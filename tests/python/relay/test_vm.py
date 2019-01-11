import tvm
import numpy as np
from tvm import relay
from tvm.relay.runtime import test_vm
from tvm.relay.backend.interpreter import TensorValue

x = relay.var('x', shape=(10, 10))
f = relay.Function([x], x)
res = test_vm(f, [TensorValue(np.random.rand(10, 10).astype('float64'))])
import pdb; pdb.set_trace()

