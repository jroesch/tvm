from __future__ import annotations
import tvm
from tvm.relay.base import Id
from tvm.relax import expr, r2

from typing import TypeVar, Generic, Union
from io import StringIO
import numpy

@r2
def add(x: Tensor, y: Tensor) -> Tensor:
    out = broadcast_shape(x.shape, y.shape)
    return compute(out, lambda indicies: x[indicies] + y[indicies])

@r2
def main(x: Tensor[10], y: Tensor[10]) -> Tensor:
    return add(x, y)

x = tvm.nd.array(numpy.random.rand(10).astype('float32'))
y = tvm.nd.array(numpy.random.rand(10).astype('float32'))
result = main(x, y)
numpy.testing.assert_allclose(result.asnumpy(), x.asnumpy() + y.asnumpy())

# @r2
# def add(x: Tensor, y: Tensor) -> Tensor:
#     xp, yp = broadcast(x, y)
#     xpp = reshape(xp, shape=product(xp.shape))
#     ypp = reshape(yp, shape=product(yp.shape))
#     out = compute(xpp.shape, lambda indices: x[indices] + y[indices])
#     return reshape(xp.shape, out)

# Tensor::FromNDArray: NDArray -> Tensor
# Tensor::ToNDArray: Tensor -> NDArray
# Tensor::Compute : TensorInfo -> ComputeRule -> fn (...) -> Tensor
# Tenosr::Scatter ...
# Tensor::Gather ...
# Tensor::Reduce

# # forall x: Tensor[n, m], y: Tensor[m] -> n == 1 => Tensor[n, m] \/ k == 3 => Tensor[k]
# @r2
# def add(x: Tensor[n, m], y: Tensor[m]) -> Tensor:
#     xp, yp = broadcast(x, y) // [n, m]
#     sh1 = product(xp.shape) // (n * m)
#     sh2 = product(yp.shape) // (n * m)
#     xpp = reshape(xp, shape=sh1) // (n * m)
#     ypp = reshape(yp, shape=sh2) // (n * m)
#     out = compute(xpp.shape, lambda indices: x[indices] + y[indices]) // (n * m)
#     if !cond:
#         thror err
#     return reshape(out, xp.shape) [n, m]

# if x.rank == 0 { if y.rank == 0 else if y.rank ==  1 { .... } } else if x.rank == 1 { ... }

# Tensor -> _|_
# k in (1, 1) => x.rank == k -> (?_k)
# x.shape[0] == 1 => ?_k == 1

# x, y = broadcast(x, y)
# add(x, y)
