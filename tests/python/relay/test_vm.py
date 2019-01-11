import tvm
import numpy as np
from tvm import relay
from tvm.relay.runtime import test_vm
from tvm.relay.backend.interpreter import TensorValue

def test_id():
    x = relay.var('x', shape=(10, 10))
    f = relay.Function([x], x)
    x_data = np.random.rand(10, 10).astype('float64')
    res = test_vm(f, [TensorValue(x_data)])
    tvm.testing.assert_allclose(res.asnumpy(), x_data)

if __name__ == "__main__":
    test_id()

