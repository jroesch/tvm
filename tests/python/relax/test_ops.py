"""Test suite for Relay Next (Relax)"""
from __future__ import annotations
import tvm
from tvm.relay.base import Id
import tvm.relax.op.operators
from tvm.relax import expr, r2


from typing import TypeVar, Generic, Union
from io import StringIO
import numpy

@r2
def add_compute(x: Tensor, y: Tensor) -> Tensor:
    out = broadcast_shape(x.shape, y.shape)
    return compute(out, lambda indicies: x[indicies] + y[indicies])

@r2
def add_main(x: Tensor[10], y: Tensor[10]) -> Tensor:
    return add(x, y)

def test_add_main():
    x = tvm.nd.array(numpy.random.rand(10).astype('float32'))
    y = tvm.nd.array(numpy.random.rand(10).astype('float32'))
    result = add_main(x, y)
    numpy.testing.assert_allclose(result.asnumpy(), x.asnumpy() + y.asnumpy())
