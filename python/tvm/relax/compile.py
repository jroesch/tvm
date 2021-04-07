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

def type_to_shape(ty):
    if isinstance(ty.shape, _expr.Tuple):
        sh = []
        for elem in ty.shape.elements:
            sh.append(elem.expr.value)
        return sh
    else:
        raise Exception("not tuple")

class Compiler:
    def __init__(self, module, main):
        self.module = module
        self.main = main
        self.ir_module = IRModule({})

    def compile(self, execute=False):
        main_fn = self.module[self.main]
        lowered_func = self.compile_func(main_fn)
        self.ir_module.update(lowered_func)
        if execute:
            print(self.ir_module)
            return tvm.build(self.ir_module, target="llvm", name='tf')
        else:
            return self.ir_module

    def lower_compute(self, compute_exp):
        import pdb; pdb.set_trace()

    def lower_func(self, func):
        def mk_tir_function(compiler, params, func):
            def create_fn_impl(compiler, inputs, outputs, func):
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
                out_sh = irb.allocate("int32", (rank,), name="B", scope="local")
                # irb.emit(tir.call_packed("relax.broadcast_shape", input_sh1, input_sh2))
                irb.emit(tir.call_packed(
                "relax.binary_broadcast_shape_fn",
                    rank,
                    rank,
                    rank,
                    input_sh1,
                    input_sh2,
                    out_sh))

                x = tvm.te.placeholder(input_shape, name="x")
                y = tvm.te.placeholder(input_shape, name="y")
                compute_output = tvm.te.compute(output_shape, lambda i: x[i] + y[i])
                schedule = tvm.te.create_schedule([compute_output.op])
                lowered = tvm.lower(schedule, [x, y, compute_output], simple_mode=False)
                # params = lowered["main"].params
                buffer_keys, buffer_values = zip(*lowered["main"].buffer_map.items())
                buffer_values = [buffer.data for buffer in buffer_values]
                params = buffer_values
                sub_vars = [buffer.data for buffer in inputs]
                sub_map = { params[0]: sub_vars[0], params[1]: sub_vars[1], params[2]: outputs[0].data }
                kernel_impl = lowered["main"].body
                kernel_impl = substitute(kernel_impl, sub_map)
                irb.emit(kernel_impl)
                # TOOD(@jroesch): improve TIR code generator
                # gv = tvm.relay.GlobalVar("my_compute")
                # compiler.ir_module[gv] = lowered["main"]

                return irb.get()

            input_shape = (10, )
            output_shape = (10, )
            out = tvm.tir.decl_buffer(output_shape, name="output", dtype="float32")
            name = func.name


            return tvm.te.extern(
                [out.shape],
                params,
                lambda ins, outs: create_fn_impl(self, ins, outs, func),
                out_buffers=[out],
                name=name,
                tag=name,
            )

        inputs = []
        for param in func.params:
            name = param.id.name_hint

            shape = type_to_shape(param.ty)

            if param.ty.dtype is None:
                dtype = "float32"
            else:
                raise Exception("invalid datatype")

            inputs.append(tvm.te.placeholder(shape, name=name, dtype=dtype))

        outputs = [mk_tir_function(self, inputs, func)]
        schedule = tvm.topi.generic.schedule_extern(outputs)
        return tvm.lower(schedule, inputs + outputs, simple_mode=True)

    def compile_func(self, func):
        # import pdb; pdb.set_trace()
        return self.lower_func(func)
