import tvm
from tvm import relay
from tvm.relay import Kind
import numpy as np

def while_loop(cond, loop_vars, loop_bodies):
    sb = relay.ScopeBuilder()
    wl = relay.Var("while_loop")
    with sb.if_scope(cond(*loop_vars)):
        sb.ret(wl(*loop_bodies))
    with sb.else_scope():
        sb.ret(relay.Tuple(loop_vars))

    def _while_loop(*args):
        return relay.Let(
            wl, relay.Function(loop_vars, sb.get()),
            wl(*args))

    return _while_loop


def foreach(iter, init, body, state_type=None):
    i = relay.var("i", shape=(), dtype='int32')
    # If user provides more generic annotation
    # we will overload.
    if state_type is None:
        state_type = relay.TypeOf(init)
    import pdb; pdb.set_trace()
    st = relay.var("st", type_annotation=state_type)
    update = body(i, st)
    dim = relay.take(relay.op.shape_of(iter), indices=i, axis=0)
    def _cond(i, st):
        return relay.op.min(relay.op.less(i, dim))
    loop = while_loop(
        _cond, [i, st], [i + int32(1), update])
    return loop(int32(0), init)

def int32(val):
    return relay.const(val, 'int32')

def test_dyn_arange():
    m, n, k = relay.TypeVar('m', Kind.ShapeVar), relay.TypeVar('n', Kind.ShapeVar), relay.TypeVar('k', Kind.ShapeVar)
    # m, n, k = tvm.var('m'), tvm.var('n'), tvm.var('k')
    x = relay.var('x', shape=(m.var, n.var, k.var), dtype='float32')
    y0 = relay.shape_of(x)
    y1 = relay.take(y0, relay.const(0, 'int32'))
    y2 = relay.op.arange(y1)
    print(y2)
    ex = relay.create_executor()
    f = relay.Function([x], y2, type_params=[m, n, k])
    data = np.random.rand(10, 5, 3).astype('float32')
    result = ex.evaluate(f)(data)
    np.testing.assert_allclose(result.asnumpy(), np.array(range(10)))

def test_dyn_concat():
    init = relay.op.reshape(relay.const(0.0), (1,))
    iter = relay.op.arange(int32(10))

    def _body(i, st):
        i = relay.op.reshape(i.astype('float32'), (1,))
        return relay.op.concatenate([st, i], axis=0)

    res = foreach(iter, init, _body, state_type=relay.TensorType([relay.Any()]))
    tres = relay.ir_pass.infer_type(res)
    print("type check")
    print(tres.astext(True))
    import pdb; pdb.set_trace()
    """
    i = relay.var("i", shape=(), dtype='int32')
    st = relay.var("st", type_annotation=relay.TypeOf(init))
    update = _body(i, st)
    dim = relay.take(relay.op.shape_of(iter), indices=i, axis=0)
    def _cond(i, st):
        return relay.op.min(relay.op.less(i, dim))

    mod = relay.module.Module()
    wl = relay.GlobalVar("while_loop")
    loop_vars = [i, st]
    sb = relay.ScopeBuilder()
    with sb.if_scope(_cond(*loop_vars)):
        sb.ret(wl(i + int32(1), update))
    with sb.else_scope():
        sb.ret(relay.Tuple(loop_vars))
    func = relay.Function(loop_vars, sb.get())
    mod[wl] = func
    print(relay.ir_pass.infer_type(func, mod=mod))
    ret = relay.Call(wl, [relay.const(0, 'int32'), init])
    mod[mod.entry_func] = relay.Function([], ret)
    print(relay.ir_pass.infer_type(mod[mod.entry_func], mod=mod))
    """"

    ex = relay.create_executor("debug", mod=mod, ctx=tvm.cpu(), target="llvm")
    result = ex.evaluate(mod.entry_func)()
    print(result)
    # import pdb; pdb.set_trace()
    # np.testing.assert_allclose(result.asnumpy(), np.array(range(10)))


if __name__ == "__main__":
    # test_dyn_arange()
    test_dyn_concat()
