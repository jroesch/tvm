"""Roundtripping tests for Relay Next (Relax)"""
from __future__ import annotations
from os import X_OK
import tvm
from tvm.relay.base import Id
import tvm.relax.op.operators
from tvm.relax import expr, r2


from typing import TypeVar, Generic, Union
from io import StringIO
import numpy

def assert_structural_equal(lhs, rhs, map_free_vars=False):
    lhs = tvm.runtime.convert(lhs)
    rhs = tvm.runtime.convert(rhs)
    # These are packed funcs here
    tvm.runtime._ffi_node_api.StructuralEqual(lhs, rhs, True, map_free_vars)

@r2
def foo(x: Tensor) -> Tensor:
    return x

foo1 = foo

@r2
def same_as_foo(x: Tensor) -> Tensor:
    return x

@r2
def not_foo(x: Tensor, y: Tensor) -> Tensor:
    return x

@r2
def foo(y: Tensor) -> Tensor:
    return y

foo2 = foo


# test literally the same object
def test_same():
    rlx_program = foo
    assert_structural_equal(rlx_program.module['foo'], rlx_program.module['foo'])


# test two fns with the same name but different objects, different variable names
# problem with span
def test_same_name():
    assert_structural_equal(foo1.module['foo'], foo2.module['foo'], True)


# test two functions that are the same with different names
def test_same_as_foo():
    rlx_program1 = foo
    rlx_program2 = same_as_foo
    assert_structural_equal(rlx_program1.module['foo'], rlx_program2.module['same_as_foo'], True)

def test_not_foo():
    rlx_program1 = foo
    rlx_program2 = not_foo
    assert_structural_equal(rlx_program1.module['foo'], rlx_program2.module['not_foo'], True)

# Tests that should succeed
test_same()
test_same_name()
test_same_as_foo()

# Tests that should fail
# test_not_foo()