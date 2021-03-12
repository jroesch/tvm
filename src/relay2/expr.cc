/* 
* Licensed to the Apache Software Foundation (ASF) under one
* or more contributor license agreements.  See the NOTICE file
* distributed with this work for additional information
* regarding copyright ownership.  The ASF licenses this file
* to you under the Apache License, Version 2.0 (the
* "License"); you may not use this file except in compliance
* with the License.  You may obtain a copy of the License at
* 
*    http://www.apache.org/licenses/LICENSE-2.0
* 
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an
* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
* KIND, either express or implied.  See the License for the
* specific language governing permissions and limitations
* under the License.
*/

#include <tvm/ir/span.h>
#include <tvm/ir/type.h>
#include <tvm/node/node.h>
#include <tvm/runtime/container.h>
#include <tvm/runtime/object.h>
#include <tvm/relay/expr.h>
#include "/home/jroesch/Git/tvm/include/relay2/expr.h"

namespace tvm { 
namespace relay2 { 
namespace expr { 

Type::Type(
    Span span) {
    ObjectPtr<TypeNode> n = make_object<TypeNode>();
    n->span = std::move(span);
    data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(TypeNode);

TVM_REGISTER_GLOBAL("relay2.Type").set_body_typed([](Span span) {
    return Type(span);
});

Expr::Expr(
    Span span) {
    ObjectPtr<ExprNode> n = make_object<ExprNode>();
    n->span = std::move(span);
    data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(ExprNode);

TVM_REGISTER_GLOBAL("relay2.Expr").set_body_typed([](Span span) {
    return Expr(span);
});

Var::Var(
    Optional<relay::Id> id,
    Type ty,
    Span span) {
    ObjectPtr<VarNode> n = make_object<VarNode>();
    n->id = std::move(id);
    n->ty = std::move(ty);
    n->span = std::move(span);
    data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(VarNode);

TVM_REGISTER_GLOBAL("relay2.Var").set_body_typed([](Optional<relay::Id> id,Type ty,Span span) {
    return Var(id,ty,span);
});

Binding::Binding(
    Var var,
    Expr val) {
    ObjectPtr<BindingNode> n = make_object<BindingNode>();
    n->var = std::move(var);
    n->val = std::move(val);
    data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(BindingNode);

TVM_REGISTER_GLOBAL("relay2.Binding").set_body_typed([](Var var,Expr val) {
    return Binding(var,val);
});

Let::Let(
    runtime::Array<Binding> bindings,
    Expr body,
    Span span) {
    ObjectPtr<LetNode> n = make_object<LetNode>();
    n->bindings = std::move(bindings);
    n->body = std::move(body);
    n->span = std::move(span);
    data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(LetNode);

TVM_REGISTER_GLOBAL("relay2.Let").set_body_typed([](runtime::Array<Binding> bindings,Expr body,Span span) {
    return Let(bindings,body,span);
});

Call::Call(
    Expr fn,
    runtime::Array<Expr> args,
    Span span) {
    ObjectPtr<CallNode> n = make_object<CallNode>();
    n->fn = std::move(fn);
    n->args = std::move(args);
    n->span = std::move(span);
    data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(CallNode);

TVM_REGISTER_GLOBAL("relay2.Call").set_body_typed([](Expr fn,runtime::Array<Expr> args,Span span) {
    return Call(fn,args,span);
});

Function::Function(
    Optional<runtime::String> name,
    runtime::Array<Var> params,
    Expr body,
    Type ret_type,
    Span span) {
    ObjectPtr<FunctionNode> n = make_object<FunctionNode>();
    n->name = std::move(name);
    n->params = std::move(params);
    n->body = std::move(body);
    n->ret_type = std::move(ret_type);
    n->span = std::move(span);
    data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(FunctionNode);

TVM_REGISTER_GLOBAL("relay2.Function").set_body_typed([](Optional<runtime::String> name,runtime::Array<Var> params,Expr body,Type ret_type,Span span) {
    return Function(name,params,body,ret_type,span);
});

BroadcastShape::BroadcastShape(
    Expr lhs,
    Expr rhs,
    Span span) {
    ObjectPtr<BroadcastShapeNode> n = make_object<BroadcastShapeNode>();
    n->lhs = std::move(lhs);
    n->rhs = std::move(rhs);
    n->span = std::move(span);
    data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(BroadcastShapeNode);

TVM_REGISTER_GLOBAL("relay2.BroadcastShape").set_body_typed([](Expr lhs,Expr rhs,Span span) {
    return BroadcastShape(lhs,rhs,span);
});

ShapeOf::ShapeOf(
    Expr tensor,
    Span span) {
    ObjectPtr<ShapeOfNode> n = make_object<ShapeOfNode>();
    n->tensor = std::move(tensor);
    n->span = std::move(span);
    data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(ShapeOfNode);

TVM_REGISTER_GLOBAL("relay2.ShapeOf").set_body_typed([](Expr tensor,Span span) {
    return ShapeOf(tensor,span);
});

TensorSlice::TensorSlice(
    Expr tensor,
    Array<Expr> slice,
    Span span) {
    ObjectPtr<TensorSliceNode> n = make_object<TensorSliceNode>();
    n->tensor = std::move(tensor);
    n->slice = std::move(slice);
    n->span = std::move(span);
    data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(TensorSliceNode);

TVM_REGISTER_GLOBAL("relay2.TensorSlice").set_body_typed([](Expr tensor,Array<Expr> slice,Span span) {
    return TensorSlice(tensor,slice,span);
});

Compute::Compute(
    Expr out_shape,
    Expr compute_body,
    Span span) {
    ObjectPtr<ComputeNode> n = make_object<ComputeNode>();
    n->out_shape = std::move(out_shape);
    n->compute_body = std::move(compute_body);
    n->span = std::move(span);
    data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(ComputeNode);

TVM_REGISTER_GLOBAL("relay2.Compute").set_body_typed([](Expr out_shape,Expr compute_body,Span span) {
    return Compute(out_shape,compute_body,span);
});

Add::Add(
    Expr lhs,
    Expr rhs,
    Span span) {
    ObjectPtr<AddNode> n = make_object<AddNode>();
    n->lhs = std::move(lhs);
    n->rhs = std::move(rhs);
    n->span = std::move(span);
    data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(AddNode);

TVM_REGISTER_GLOBAL("relay2.Add").set_body_typed([](Expr lhs,Expr rhs,Span span) {
    return Add(lhs,rhs,span);
});

Dim::Dim(
    Span span) {
    ObjectPtr<DimNode> n = make_object<DimNode>();
    n->span = std::move(span);
    data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(DimNode);

TVM_REGISTER_GLOBAL("relay2.Dim").set_body_typed([](Span span) {
    return Dim(span);
});

Shape::Shape(
    Span span) {
    ObjectPtr<ShapeNode> n = make_object<ShapeNode>();
    n->span = std::move(span);
    data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(ShapeNode);

TVM_REGISTER_GLOBAL("relay2.Shape").set_body_typed([](Span span) {
    return Shape(span);
});

Tensor::Tensor(
    Optional<Expr> shape,
    Optional<Expr> dtype,
    Span span) {
    ObjectPtr<TensorNode> n = make_object<TensorNode>();
    n->shape = std::move(shape);
    n->dtype = std::move(dtype);
    n->span = std::move(span);
    data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(TensorNode);

TVM_REGISTER_GLOBAL("relay2.Tensor").set_body_typed([](Optional<Expr> shape,Optional<Expr> dtype,Span span) {
    return Tensor(shape,dtype,span);
});

} // namespace expr 
} // namespace relay2 
} // namespace tvm 
