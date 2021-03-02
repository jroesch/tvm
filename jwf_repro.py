import onnx
import tvm
from tvm import relay
import numpy as np
from tvm.contrib import graph_runtime

onnx_model = onnx.load('resnet34-ssd1200.onnx')
input_name = "image"
input_shape = [1, 3, 1200, 1200]
shape_dict = {input_name: input_shape}
dtype='float32'

mod, params = relay.frontend.from_onnx(onnx_model, shape=shape_dict, freeze_params=True)
main_fn = mod["main"]
mod["main"] = relay.Function(main_fn.params, main_fn.body[0], main_fn.ret_type, main_fn.type_params, main_fn.attrs)
# fix this?
# mod = relay.transform.AnnotateSpans()(mod)

print("Compile")
target = {"gpu": "vulkan", "cpu": "llvm"}
ctx = [tvm.vulkan(), tvm.cpu()]
config = {"relay.fallback_device_type": tvm.vulkan().device_type}
with tvm.transform.PassContext(opt_level=4, config=config):
    vm_exec = relay.vm.compile(mod, target, params=params)

print("Inference")
# Run inference on sample data
x = np.random.normal(size=input_shape).astype(dtype)
vm = tvm.runtime.vm.VirtualMachine(vm_exec, ctx)
input_dict = {input_name: x}

#input_dict = BERT_INPUT_DICT

vm.set_input("main", **input_dict)  # required
tvm_output = vm.run()
print(tvm_output[0])
