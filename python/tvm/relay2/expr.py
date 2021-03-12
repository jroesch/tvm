# 
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
# 
#    http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import tvm._ffi
from ..ir.base import Node
from . import _ffi_api

ObjectRef = Node
@tvm._ffi.register_object("relay2.expr.Type")
class Type(ObjectRef):
    def __init__(self, span):
        self.__init_handle_by_constructor__(_ffi_api.Type,  span)


@tvm._ffi.register_object("relay2.expr.Expr")
class Expr(ObjectRef):
    def __init__(self, span):
        self.__init_handle_by_constructor__(_ffi_api.Expr,  span)


@tvm._ffi.register_object("relay2.expr.Var")
class Var(Expr):
    def __init__(self, id, ty, span):
        self.__init_handle_by_constructor__(_ffi_api.Var,  id,  ty,  span)


@tvm._ffi.register_object("relay2.expr.Binding")
class Binding(ObjectRef):
    def __init__(self, var, val):
        self.__init_handle_by_constructor__(_ffi_api.Binding,  var,  val)


@tvm._ffi.register_object("relay2.expr.Let")
class Let(Expr):
    def __init__(self, bindings, body, span):
        self.__init_handle_by_constructor__(_ffi_api.Let,  bindings,  body,  span)


@tvm._ffi.register_object("relay2.expr.Call")
class Call(Expr):
    def __init__(self, fn, args, span):
        self.__init_handle_by_constructor__(_ffi_api.Call,  fn,  args,  span)


@tvm._ffi.register_object("relay2.expr.Function")
class Function(Expr):
    def __init__(self, name, params, body, ret_type, span):
        self.__init_handle_by_constructor__(_ffi_api.Function,  name,  params,  body,  ret_type,  span)


@tvm._ffi.register_object("relay2.expr.BroadcastShape")
class BroadcastShape(Expr):
    def __init__(self, lhs, rhs, span):
        self.__init_handle_by_constructor__(_ffi_api.BroadcastShape,  lhs,  rhs,  span)


@tvm._ffi.register_object("relay2.expr.ShapeOf")
class ShapeOf(Expr):
    def __init__(self, tensor, span):
        self.__init_handle_by_constructor__(_ffi_api.ShapeOf,  tensor,  span)


@tvm._ffi.register_object("relay2.expr.TensorSlice")
class TensorSlice(Expr):
    def __init__(self, tensor, slice, span):
        self.__init_handle_by_constructor__(_ffi_api.TensorSlice,  tensor,  slice,  span)


@tvm._ffi.register_object("relay2.expr.Compute")
class Compute(Expr):
    def __init__(self, out_shape, compute_body, span):
        self.__init_handle_by_constructor__(_ffi_api.Compute,  out_shape,  compute_body,  span)


@tvm._ffi.register_object("relay2.expr.Add")
class Add(Expr):
    def __init__(self, lhs, rhs, span):
        self.__init_handle_by_constructor__(_ffi_api.Add,  lhs,  rhs,  span)


@tvm._ffi.register_object("relay2.expr.Dim")
class Dim(Type):
    def __init__(self, span):
        self.__init_handle_by_constructor__(_ffi_api.Dim,  span)


@tvm._ffi.register_object("relay2.expr.Shape")
class Shape(Type):
    def __init__(self, span):
        self.__init_handle_by_constructor__(_ffi_api.Shape,  span)


@tvm._ffi.register_object("relay2.expr.Tensor")
class Tensor(Type):
    def __init__(self, shape, dtype, span):
        self.__init_handle_by_constructor__(_ffi_api.Tensor,  shape,  dtype,  span)


