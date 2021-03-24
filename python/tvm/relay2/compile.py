import tvm
from tvm import tir, IRModule
from tvm.driver.build_module import lower, build
from tvm.tir import ir_builder


@tvm.register_func("relay2.broadcast_shape")
def broadcast_shape(*inputs):
    import pdb; pdb.set_trace()

class ExprMutator:
    def visit(self, expr):
        if isinstance(expr, expr.Var):
            return self.visit_var(expr)
        elif isinstance(expr, expr.GlobalVar):
            return self.visit_global_var(expr)
        elif isinstance(expr, expr.Let):
            return self.visit_let(expr)
        elif isinstance(expr, expr.Call):
            return self.visit_call(expr)
        elif isinstance(expr, expr.Function):
            return self.visit_func(expr)
        elif isinstance(expr, expr.BroadcastShape):
            return self.visit_bs(expr)
        elif isinstance(expr, expr.ShapeOf):
            return self.visit_shape_of(expr)
        elif isinstance(expr, expr.TensorSlice):
            return self.visit_tensor_slice(expr)
        elif isinstance(expr, expr.Compute):
            return self.visit_compute(expr)
        elif isinstance(expr, expr.Tuple):
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
        def the_function(*inputs):
            def create_fn_impl(inputs, outputs):
                irb = ir_builder.create()
                assert len(inputs), 1
                assert len(outputs), 1
                # import pdb; pdb.set_trace()
                # irb_params = []
                # for param in func.params:
                #     buffer = tvm.tir.decl_buffer((10,), name=param.id.name_hint, dtype="float32")
                #     irb_params.append(self.irb.buffer_ptr(buffer)
                irb.emit(tir.call_packed("relay2.broadcast_shape", inputs[0]))

                return irb.get()


            out = tvm.tir.decl_buffer((10,), name="output", dtype="float32")
            name = func.name

            return tvm.te.extern(
                [out.shape],
                inputs,
                lambda ins, outs: create_fn_impl(ins, outs),
                out_buffers=[out],
                name=name,
                tag=name,
            )

        the_input = tvm.te.placeholder((10, ), name="input", dtype="float32")
        the_output = the_function(the_input)
        schedule = tvm.topi.generic.schedule_extern([the_output])
        return tvm.lower(schedule, [the_input, the_output], simple_mode=True)

    def compile_func(self, func):
        # import pdb; pdb.set_trace()
        return self.lower_func(func)
