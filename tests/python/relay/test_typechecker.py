"""Test that type checker correcly computes types
   for expressions.
"""
from tvm.relay.type_infer import check_expr
from tvm.relay.ir_builder import IRBuilder, float_type, op, func_type
from tvm.relay.env import Environment

def has_type(expr, typ):
    env = Environment({})
    checked_expr = check_expr(env, expr)
    return checked_expr.checked_type() == typ

def test_monomorphic_let():
    "Program: let x = 1; x"
    b = IRBuilder()
    x = b.let('x', 1, value_type=float_type())
    b.ret(x)

    prog = b.get()
    assert has_type(prog, float_type())


def test_single_op():
    "Program: fn (x : float32) { let t1 = f(x); t1 }"
    b = IRBuilder()
    f = op('log')
    with b.function(('x', float_type())) as func:
        x, = func.param_ids()
        t1 = b.let('t1', f(x))
        b.ret(t1)
    assert has_type(func.to_func(), func_type([float_type()], float_type()))
