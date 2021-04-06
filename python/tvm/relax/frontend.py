from __future__ import annotations
import tvm
from tvm.relay.base import Id
from tvm.relax import expr
from tvm import tir
import numpy as np
import synr

from typing import TypeVar, Generic, Union

from synr import ast, Transformer
from synr.diagnostic_context import DiagnosticContext
from io import StringIO
from .compile import Compiler

def print_ty(ty):
    if isinstance(ty, expr.Dim):
        return "Dim"
    elif isinstance(ty, expr.Tensor):
        return "Tensor"
    else:
        return "UNKNOWN"

def print_fn(func):
    buffer = StringIO("")
    param_str = ""
    for param in func.params:
        param_str += f"{param.id.name_hint}: {print_ty(param.ty)}, "

    buffer.write(f"fn {func.name}({param_str}) {{\n")
    buffer.write(f"{func.body}\n")
    buffer.write("}")
    return buffer.getvalue()

expr.Function.__str__ = print_fn

class R2Transformer(Transformer):
    def __init__(self):
        self.str_to_var = {}
        self.blocks = []
        super().__init__()

    def decl_var(self, name, ty, span=None):
        identifier = Id(name)
        var = expr.Var(identifier, ty, span)
        self.str_to_var[name] = var
        return var

    def to_type(self, ty):
        if ty is None:
            return None

        if isinstance(ty, ast.TypeVar):
            if ty.id.name == "Tensor":
                return expr.Tensor(None, None, None)

        if isinstance(ty, ast.TypeApply):
            if ty.id.name == "Tensor":
                dims = []
                # TODO(@jroesch): add support for dtype
                for param in ty.params:
                    if isinstance(param, ast.TypeConstant):
                        dim = expr.TIRExpr(tir.IntImm("int32", param.value), None)
                        dims.append(dim)
                return expr.Tensor(expr.Tuple(dims, span=None), None, None)

        import pdb; pdb.set_trace()


        self._diagnostic_context.emit('error', "invalid type", ty.span)
        self._diagnostic_context.render()

    def transform_module(self, mod: ast.Module) -> M:
        module = {}
        for func_name in mod.funcs:
            func = mod.funcs[func_name]
            module[func_name] = self.transform_function(func)
        return module

    def transform_function(self, func: ast.Function) -> F:
        params = []
        for param in func.params:
            ty = self.to_type(param.ty)
            param = self.decl_var(param.name, ty, None)
            params.append(param)
        new_body = self.transform_block(func.body)
        return expr.Function(func.name, params, new_body, None, None)

    def transform_stmt(self, stmt: ast.Stmt) -> S:
        if isinstance(stmt, ast.Assign):
            assert isinstance(stmt.lhs, ast.Var)
            lhs = self.decl_var(stmt.lhs.id.name, None, None)
            rhs = self.transform_expr(stmt.rhs)
            self.blocks[-1].append(expr.Binding(lhs, rhs))
            return None
        elif isinstance(stmt, ast.Return):
            return self.transform_expr(stmt.value)
        else:
            self._diagnostic_context.emit('error', "only variable left-hand sides are supported in Relay", stmt.span)
            self._diagnostic_context.render()

    def transform_expr(self, exp: ast.Expr) -> E:
        if isinstance(exp, ast.Call):
            if isinstance(exp.func_name, ast.Var):
                params = []
                for arg in exp.params:
                    params.append(self.transform_expr(arg))

                if exp.func_name.id.name == "broadcast_shape":
                    if len(params) != 2:
                        self._diagnostic_context.emit('error', f"broadcast_shape only takes 2 arguments {params.len()}", exp.span)
                        self._diagnostic_context.render()
                    return expr.BroadcastShape(params[0], params[1], span=None)
                elif exp.func_name.id.name == "compute":
                    if len(params) != 2:
                        self._diagnostic_context.emit('error', f"compute only takes 2 arguments {params.len()}", exp.span)
                        self._diagnostic_context.render()
                    return expr.Compute(params[0], params[1], span=None)
                else:
                    if exp.func_name.id.name in self.str_to_var:
                        return self.str_to_var[exp.func_name.id.name]
                    else:
                        # todo: globalvar equality? use global str -> id map?
                        ident = Id(exp.func_name.id.name)
                        return expr.Call(expr.GlobalVar(ident, None, None), params, None)

                    self._diagnostic_context.emit('error', f"unknown functionc all {len(params)}", exp.span)
                    self._diagnostic_context.render()
            elif isinstance(exp.func_name, ast.Op):
                if exp.func_name.name == ast.BuiltinOp.Subscript:
                    tensor = self.transform_expr(exp.params[0])
                    indicies = []
                    for index in exp.params[1].values:
                        indicies.append(self.transform_expr(index))
                    return expr.TensorSlice(tensor, indicies, None)
                elif exp.func_name.name == ast.BuiltinOp.Add:
                    params = []
                    for arg in exp.params:
                        params.append(self.transform_expr(arg))
                    return expr.Add(params[0], params[1], None)

            self._diagnostic_context.emit('error', "unsupported function", exp.span)
            self._diagnostic_context.render()
        elif isinstance(exp, ast.Attr):
            field_name = exp.field.name
            tensor = self.transform_expr(exp.object)

            if field_name == "shape":
                return expr.ShapeOf(tensor, None)
            else:
                self._diagnostic_context.emit('error', "unsupported function", exp.span)
                self._diagnostic_context.render()
        elif isinstance(exp, ast.Function):
            print(exp)
            return self.transform_function(exp)
        elif isinstance(exp, ast.Tuple):
            assert False
        elif isinstance(exp, ast.Var):
            return self.str_to_var[exp.id.name]
        else:
            self._diagnostic_context.emit('error', f"don't support this construct {type(exp)}", exp.span)
            self._diagnostic_context.render()

    def enter_block(self):
        self.blocks.append([])

    def exit_block(self):
        back = self.blocks[-1]
        self.blocks.pop()
        return back

    def transform_block(self, block: ast.Block) -> B:
        self.enter_block()

        for stmt in block.stmts[:-1]:
            assert self.transform_stmt(stmt) is None

        ret_expr = self.transform_stmt(block.stmts[-1])
        # assert ret_expr is not None

        bindings = self.exit_block()
        return expr.Let(bindings, ret_expr, span=None)

    def transform_parameter(self, expr: ast.Parameter) -> P:
        pass

    def transform_type(self, ty: ast.Type) -> T:
        pass

GLOBAL_MODULE = {}

def add_to_global_module(new_fns):
    global GLOBAL_MODULE
    for fn_name in new_fns:
        if fn_name in GLOBAL_MODULE:
            raise Exception(f"duplicate name {fn_name}")
        else:
            GLOBAL_MODULE[fn_name] = new_fns[fn_name]


def r2(f):
    diag_cx = synr.PrinterDiagnosticContext()
    ast = synr.to_ast(f, diag_cx)
    module = R2Transformer().do_transform(ast, diag_cx)
    add_to_global_module(module)
    def __wrapper(*inputs):
        compiler = Compiler(module, f.__name__)
        compiled_f = compiler.compile(execute=True)
        # Actually compute needed buffer sizes.
        out = tvm.nd.array(np.random.rand(10).astype('float32'))
        import pdb; pdb.set_trace()
        compiled_f(*(list(inputs) + [out]))
        return out

    return __wrapper
