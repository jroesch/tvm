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

def test_op():
    x = relay.var('x', shape=(10, 10))
    f = relay.Function([x], x + x)
    x_data = np.random.rand(10, 10).astype('float32')
    res = test_vm(f, [TensorValue(x_data)])
    tvm.testing.assert_allclose(res.asnumpy(), x_data + x_data)

def any(x):
    x = relay.op.nn.batch_flatten(x)
    return relay.op.min(x, axis=[0, 1])

def test_cond():
    x = relay.var('x', shape=(10, 10))
    y = relay.var('x', shape=(10, 10))
    # f = relay.Function([x, y], relay.op.equal(x, y))
    f = relay.Function([x, y], any(relay.op.equal(x, y)))
    x_data = np.random.rand(10, 10).astype('float32')
    y_data = np.random.rand(10, 10).astype('float32')

    # same
    res = test_vm(f, [TensorValue(x_data), TensorValue(x_data)])
    tvm.testing.assert_allclose(res.asnumpy(), True)

    # diff
    res = test_vm(f, [TensorValue(x_data), TensorValue(y_data)])
    tvm.testing.assert_allclose(res.asnumpy(), False)


def test_simple_if():
    x = relay.var('x', shape=(10, 10))
    y = relay.var('x', shape=(10, 10))
    f = relay.Function([x, y],
        relay.If(any(relay.op.equal(x, y)), x, y))
    x_data = np.random.rand(10, 10).astype('float32')
    y_data = np.random.rand(10, 10).astype('float32')

    # same
    res = test_vm(f, [TensorValue(x_data), TensorValue(x_data)])
    tvm.testing.assert_allclose(res.asnumpy(), x_data)

    # diff
    res = test_vm(f, [TensorValue(x_data), TensorValue(y_data)])
    tvm.testing.assert_allclose(res.asnumpy(), y_data)


if __name__ == "__main__":
    test_id()
    test_op()
    test_cond()
    test_simple_if()

