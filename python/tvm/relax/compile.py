from typing import Optional, Union, List
import attr

import tvm
from tvm import tir, IRModule
from tvm.driver.build_module import lower, build
from tvm.tir import ir_builder
from tvm.relax import expr as _expr
from tvm.tir.stmt_functor import substitute


@tvm.register_func("relax.broadcast_shape")
def broadcast_shape(*inputs):
    import pdb; pdb.set_trace()

class ExprMutator:
    def visit(self, expr):
        if isinstance(expr, _expr.Var):
            return self.visit_var(expr)
        elif isinstance(expr, _expr.GlobalVar):
            return self.visit_global_var(expr)
        elif isinstance(expr, _expr.Let):
            return self.visit_let(expr)
        elif isinstance(expr, _expr.Call):
            return self.visit_call(expr)
        elif isinstance(expr, _expr.Function):
            return self.visit_func(expr)
        elif isinstance(expr, _expr.BroadcastShape):
            return self.visit_bs(expr)
        elif isinstance(expr, _expr.ShapeOf):
            return self.visit_shape_of(expr)
        elif isinstance(expr, _expr.TensorSlice):
            return self.visit_tensor_slice(expr)
        elif isinstance(expr, _expr.Compute):
            return self.visit_compute(expr)
        elif isinstance(expr, _expr.Tuple):
            return self.visit_tuple(expr)
        else:
            assert False

    def visit_var(self, var):
        return var

    def visit_global_var(self, var):
        return var

    def visit_let(self, let):
        assert False

class Specializer(ExprMutator):
    def __init__(self):
        self.specialized = {}

    def specialize(self, function, shape_info):
        import pdb; pdb.set_trace()

def type_to_shape(diag_cx, ty):
    if isinstance(ty.shape, _expr.Tuple):
        sh = []
        for elem in ty.shape.elements:
            sh.append(elem.expr.value)
        return sh
    else:
        return None
        # diag_cx.emit('error', f"unspecified rank, and/or dimensions on tensor type found: {ty.shape}", ty.span)
        # diag_cx.render()

def inline_tir(args, lowered_func):
    buffer_keys, buffer_values = zip(*lowered_func.buffer_map.items())
    buffer_values = [buffer.data for buffer in buffer_values]
    sub_vars = [buffer.data for buffer in args]
    subs = {}
    for value, sub_var in zip(buffer_values, sub_vars):
        subs[value] = sub_var
    kernel_impl = lowered_func.body
    kernel_impl = substitute(kernel_impl, subs)
    return kernel_impl

@attr.s(auto_attribs=True)
class Parameter:
    data: tvm.tir.Var
    rank: Union[tvm.tir.Var, int]
    shape: Optional[List[tvm.tir.Var]]

def create_fn_impl(compiler, inputs, outputs, func):
    input_shape = (10, )
    output_shape = (10, )
    irb = ir_builder.create()
    # import pdb; pdb.set_trace()
    # irb_params = []
    # for param in func.params:
    #     buffer = tvm.tir.decl_buffer((10,), name=param.id.name_hint, dtype="float32")
    #     irb_params.append(self.irb.buffer_ptr(buffer)
    rank = 1
    input_sh1 = irb.allocate("int32", (rank,), name="A", scope="local")
    input_sh2 = irb.allocate("int32", (rank,), name="B", scope="local")
    input_sh1[0] = 10
    input_sh2[0] = 10
    out_sh = irb.allocate("int32", (rank,), name="C", scope="local")
    # irb.emit(tir.call_packed("relax.broadcast_shape", input_sh1, input_sh2))
    irb.emit(tir.call_packed(
    "relax.binary_broadcast_shape_fn",
        rank,
        rank,
        rank,
        input_sh1,
        input_sh2,
        out_sh))

    irb.emit(tir.call_packed(
        "relax.get_rank",
        inputs[0]))

    x = tvm.te.placeholder(input_shape, name="x")
    y = tvm.te.placeholder(input_shape, name="y")
    compute_output = tvm.te.compute(output_shape, lambda i: x[i] + y[i])
    schedule = tvm.te.create_schedule([compute_output.op])
    lowered = tvm.lower(schedule, [x, y, compute_output], simple_mode=False)
    kernel_body = inline_tir(inputs + outputs, lowered["main"])
    irb.emit(kernel_body)
    # TOOD(@jroesch): improve TIR code generator
    # gv = tvm.relay.GlobalVar("my_compute")
    # compiler.ir_module[gv] = lowered["main"]

    return irb.get()

def mk_dynamic_rank(no_params, compute_rule):
    fn_params = []

    for i in range(no_params):
        fn_params.append(tvm.tir.Var(f"param{i}", "handle"))

    tvm.tir.PrimFunc(fn_params, body)
    import pdb; pdb.set_trace()

def mk_tir_function(compiler, params, func):
    # import pdb; pdb.set_trace()
    input_shape = (10, )
    output_shape = (10, )
    out = tvm.tir.decl_buffer(output_shape, name="output", dtype="float32")
    name = func.name

    return tvm.te.extern(
        [out.shape],
        params,
        lambda ins, outs: create_fn_impl(compiler, ins, outs, func),
        out_buffers=[out],
        name=name,
        tag=name,
    )

class Compiler:
    def __init__(self, diag_cx, module, main):
        self.diag_cx = diag_cx
        self.module = module
        self.main = main
        self.ir_module = IRModule({})

    def compile(self, execute=False):
        for fn_name in self.module:
            func = self.module[fn_name]
            lowered_func = self.compile_func(func)
            print(lowered_func)
            self.ir_module.update(lowered_func)

        if execute:
            print(self.ir_module)
            return tvm.build(self.ir_module, target="llvm", name='tf')
        else:
            return self.ir_module

    def lower_compute(self, compute_exp):
        import pdb; pdb.set_trace()

    def lower_func(self, func):
        inputs = []
        for param in func.params:
            name = param.id.name_hint

            shape = type_to_shape(self.diag_cx, param.ty)

            if param.ty.dtype is None:
                dtype = "float32"
            else:
                raise Exception("invalid datatype")

            # Detect passing mode
            # Dynamic Rank
            # if shape is None:
            #     length = tvm.te.var('k')
            #     data = tvm.te.placeholder(length, name=name, dtype=dtype)
            # else:
            inputs.append(tvm.te.placeholder((10,), name=name, dtype=dtype))

        outputs = [mk_tir_function(self, inputs, func)]
        schedule = tvm.topi.generic.schedule_extern(outputs)
        return tvm.lower(schedule, inputs + outputs, simple_mode=True)

    def compile_func(self, func):
        # import pdb; pdb.set_trace()
        return self.lower_func(func)
