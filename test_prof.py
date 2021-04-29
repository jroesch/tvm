import numpy as np
import tvm
from tvm.relay.testing import resnet


module, params = resnet.get_workload()

with tvm.transform.PassContext(opt_level=3):
    lib = tvm.relay.build(module, target="llvm", params=params)
    # f = tvm.relay.create_executor('graph', mod=module).evaluate()
    # input_image = np.random.rand(1, 3, 224, 224)
    # out = f(input_image, **params)
