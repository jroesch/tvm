import tvm
import numpy as np
from tvm import relay
from tvm.relay.vm import eval_vm
from tvm.relay.scope_builder import ScopeBuilder

def test_id():
    x = relay.var('x', shape=(10, 10))
    f = relay.Function([x], x)
    x_data = np.random.rand(10, 10).astype('float64')
    res = eval_vm(f, x_data)
    tvm.testing.assert_allclose(res.asnumpy(), x_data)

def test_op():
    x = relay.var('x', shape=(10, 10))
    f = relay.Function([x], x + x)
    x_data = np.random.rand(10, 10).astype('float32')
    res = eval_vm(f, x_data)
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
    res = eval_vm(f, x_data, x_data)
    tvm.testing.assert_allclose(res.asnumpy(), True)

    # diff
    res = eval_vm(f, x_data, y_data)
    tvm.testing.assert_allclose(res.asnumpy(), False)


def test_simple_if():
    x = relay.var('x', shape=(10, 10))
    y = relay.var('x', shape=(10, 10))
    f = relay.Function([x, y],
        relay.If(any(relay.op.equal(x, y)), x, y))
    x_data = np.random.rand(10, 10).astype('float32')
    y_data = np.random.rand(10, 10).astype('float32')

    # same
    res = eval_vm(f, x_data, x_data)
    tvm.testing.assert_allclose(res.asnumpy(), x_data)

    # diff
    res = eval_vm(f, x_data, y_data)
    tvm.testing.assert_allclose(res.asnumpy(), y_data)

def test_simple_call():
    mod = relay.module.Module({})
    sum_up = relay.GlobalVar('sum_up')
    i = relay.var('i', shape=[], dtype='int32')
    sb = ScopeBuilder()
    sb.ret(i)
    func = relay.Function([i], sb.get(), ret_type=relay.TensorType([], 'int32'))
    mod[sum_up] = func
    i_data = np.array(0, dtype='int32')
    result = eval_vm(sum_up, i_data, mod=mod)
    tvm.testing.assert_allclose(result.asnumpy(), i_data)

def test_count_loop():
    mod = relay.module.Module({})
    sum_up = relay.GlobalVar('sum_up')
    i = relay.var('i', shape=[], dtype='int32')
    sb = ScopeBuilder()
    with sb.if_scope(relay.equal(i, relay.const(0, dtype='int32'))):
        sb.ret(i)
    with sb.else_scope():
        one_less = relay.subtract(i, relay.const(1, dtype='int32'))
        rec_call = relay.Call(sum_up, [one_less])
        sb.ret(relay.add(rec_call, i))
    func = relay.Function([i], sb.get(), ret_type=relay.TensorType([], 'int32'))
    mod[sum_up] = func
    i_data = np.array(0, dtype='int32')
    result = eval_vm(sum_up, i_data, mod=mod)
    tvm.testing.assert_allclose(result.asnumpy(), i_data)

def test_sum_loop():
    mod = relay.module.Module({})
    sum_up = relay.GlobalVar('sum_up')
    i = relay.var('i', shape=[], dtype='int32')
    accum = relay.var('accum', shape=[], dtype='int32')
    sb = ScopeBuilder()
    with sb.if_scope(relay.equal(i, relay.const(0, 'int32'))):
        sb.ret(accum)
    with sb.else_scope():
        one_less = relay.subtract(i, relay.const(1, 'int32'))
        new_accum = relay.add(accum, i)
        sb.ret(relay.Call(sum_up, [one_less, new_accum]))
    func = relay.Function([i, accum], sb.get())
    mod[sum_up] = func
    i_data = np.array(10, dtype='int32')
    accum_data = np.array(0, dtype='int32')
    result = eval_vm(sum_up, i_data, accum_data, mod=mod)
    tvm.testing.assert_allclose(result.asnumpy(), sum(range(1, 11)))

def test_tuple_fst():
    ttype = relay.TupleType([relay.TensorType(1,), relay.TensorType(10,)])
    tup = relay.var('tup', type_annotation=ttype)
    f = relay.Function([tup], relay.TupleGetItem(tup, 0))
    i_data = np.array(0, dtype='int32')
    result = eval_vm(sum_up, i_data, mod=mod)
    tvm.testing.assert_allclose(result.asnumpy(), i_data)

if __name__ == "__main__":
    test_id()
    test_op()
    test_cond()
    test_simple_if()
    test_simple_call()
    test_count_loop()
    test_sum_loop()
