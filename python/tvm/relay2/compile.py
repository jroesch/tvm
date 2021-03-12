import tvm
from tvm.driver.build_module import lower, build
from tvm.tir import ir_builder

def broadcast_shape(x, y):
    import pdb; pdb.set_trace()

class Compiler:
    def __init__(self, module):
        self.module = module
        self.irb = None

    def compile(self):
        for fn_name in self.module:
            self.compile_func(self.module[fn_name])

    def lower_compute(self, compute_exp):
        import pdb; pdb.set_trace()

    def compile_func(self, func):
        self.irb = ir_builder.create()
        irb_params = []
        for param in func.params:
            buffer = tvm.tir.decl_buffer((10,), name=param.id.name_hint, dtype="float32")
            irb_params.append(self.irb.buffer_ptr(buffer))

        # ib.emit(tir.call_packed)

        out = tvm.tir.decl_buffer((10,), name="output", dtype="float32")

        return tvm.te.extern(
            [out.shape],
            [gen],
            lambda ins, outs: gen_ir(ins[0], outs[0], outs[1]),
        out_buffers=[out_left, out_right],
        name="threefry_split",
        tag="threefry_split",
    )
        import pdb; pdb.set_trace()
            # import pdb; pdb.set_trace()
            # gen = irb.buffer_ptr(gen_ptr)
            # out_left = irb.buffer_ptr(out_left_ptr)
            # out_right = irb.buffer_ptr(out_right_ptr

        # import pdb; pdb.set_trace()
        # gen = irb.buffer_ptr(gen_ptr)
        # out_left = irb.buffer_ptr(out_left_ptr)
        # out_right = irb.buffer_ptr(out_right_ptr
