from __future__ import annotations
from tvm.relay.base import Id
from tvm.relay2 import expr

import synr

from typing import TypeVar, Generic, Union

from synr import ast, Transformer
from synr.diagnostic_context import DiagnosticContext
from io import StringIO

def print_ty(ty):
    if isinstance(ty, expr.Dim):
        return "Dim"
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

class R2Transformer(Transformer):
    def __init__(self):
        self.str_to_id = {}
        self.blocks = []
        super().__init__()

    def transform_module(self, mod: ast.Module) -> M:
        module = {}
        for func_name in mod.funcs:
            func = mod.funcs[func_name]
            module[func_name] = self.transform_function(func)
        return module

    def transform_function(self, func: ast.Function) -> F:
        params = []
        for param in func.params:
            identifier = Id(param.name)
            self.str_to_id[param.name] = identifier
            param = expr.Var(identifier, expr.Dim(None), None)
            params.append(param)
        new_body = self.transform_block(func.body)
        return expr.Function(func.name, params, new_body, None, None)

    def transform_stmt(self, stmt: ast.Stmt) -> S:
        if isinstance(stmt, ast.Assign):
            assert isinstance(stmt.lhs, ast.Var)
            identifier = Id(stmt.lhs.id.name)
            self.str_to_id[stmt.lhs.id.name] = identifier
            lhs = expr.Var(identifier, None, None)
            rhs = self.transform_expr(stmt.rhs)
            self.blocks[-1].append(expr.Binding(lhs, rhs))
            return None
        else:
            self.diag_cx.emit('error', "only variable left-hand sides are supported in Relay", stmt.span)
            self.diag_cx.render()
            import pdb; pdb.set_trace()

    def transform_expr(self, expr: ast.Expr) -> E:
        pass

    def enter_block(self):
        self.blocks.append([])

    def exit_block(self):
        back = self.blocks[:-1]
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

def r2(f):
    diag_cx = synr.PrinterDiagnosticContext()
    ast = synr.to_ast(f, diag_cx)
    updated_ast = R2Transformer().do_transform(ast, diag_cx)
    print(print_fn(updated_ast["add"]))
    import pdb; pdb.set_trace()

@r2
def add(x: Tensor, y: Tensor) -> Tensor:
    out = broadcast_shape(x.shape, y.shape)
    compute(out, lambda indicies: x[indicies], y[indicies])

@r2
def stack(tl: Array[Tensor], dim: Dim) -> Tensor:
    return tl
