from __future__ import annotations
from tvm.relay.base import Id
from tvm.relay2 import expr, r2

from typing import TypeVar, Generic, Union
from io import StringIO

@r2
def add(x: Tensor, y: Tensor) -> Tensor:
    out = broadcast_shape(x.shape, y.shape)
    return compute(out, lambda indicies: x[indicies] + y[indicies])

# @r2
# def stack(tl: Array[Tensor], dim: Dim) -> Tensor:
#     return tl
