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
#ifndef TVM_RELAY2_EXPR_H_
#define TVM_RELAY2_EXPR_H_

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

class Type;
class TypeNode : public Object {
 public:
    Span span;
    void VisitAttrs(AttrVisitor* v) {
        v->Visit("span", &span);
    }
    bool SEqualReduce(const TypeNode* other, SEqualReducer equal) const {
        return equal(span, other->span);
    }
    void SHashReduce(SHashReducer hash_reduce) const {
        hash_reduce(span);
    }
    static constexpr const char* _type_key = "relay2.expr.Type";
    static constexpr const bool _type_has_method_sequal_reduce = true;
    static constexpr const bool _type_has_method_shash_reduce = true;
    TVM_DECLARE_BASE_OBJECT_INFO(TypeNode, Object);
};

class Type : public ObjectRef {
 public:
    TVM_DLL Type(
        Span span    );
    TVM_DEFINE_OBJECT_REF_METHODS(Type, ObjectRef, TypeNode);
};

class Expr;
class ExprNode : public Object {
 public:
    Span span;
    void VisitAttrs(AttrVisitor* v) {
        v->Visit("span", &span);
    }
    bool SEqualReduce(const ExprNode* other, SEqualReducer equal) const {
        return equal(span, other->span);
    }
    void SHashReduce(SHashReducer hash_reduce) const {
        hash_reduce(span);
    }
    static constexpr const char* _type_key = "relay2.expr.Expr";
    static constexpr const bool _type_has_method_sequal_reduce = true;
    static constexpr const bool _type_has_method_shash_reduce = true;
    TVM_DECLARE_BASE_OBJECT_INFO(ExprNode, Object);
};

class Expr : public ObjectRef {
 public:
    TVM_DLL Expr(
        Span span    );
    TVM_DEFINE_OBJECT_REF_METHODS(Expr, ObjectRef, ExprNode);
};

class Var;
class VarNode : public ExprNode {
 public:
    Optional<relay::Id> id;
    Type ty;
    Span span;
    void VisitAttrs(AttrVisitor* v) {
        v->Visit("id", &id);
        v->Visit("ty", &ty);
        v->Visit("span", &span);
    }
    bool SEqualReduce(const VarNode* other, SEqualReducer equal) const {
        return equal(id, other->id) &&  equal(ty, other->ty) &&  equal(span, other->span);
    }
    void SHashReduce(SHashReducer hash_reduce) const {
        hash_reduce(id);
        hash_reduce(ty);
        hash_reduce(span);
    }
    static constexpr const char* _type_key = "relay2.expr.Var";
    static constexpr const bool _type_has_method_sequal_reduce = true;
    static constexpr const bool _type_has_method_shash_reduce = true;
    TVM_DECLARE_FINAL_OBJECT_INFO(VarNode, ExprNode);
};

class Var : public Expr {
 public:
    TVM_DLL Var(
        Optional<relay::Id> id,
        Type ty,
        Span span    );
    TVM_DEFINE_OBJECT_REF_METHODS(Var, Expr, VarNode);
};

class Binding;
class BindingNode : public Object {
 public:
    Var var;
    Expr val;
    void VisitAttrs(AttrVisitor* v) {
        v->Visit("var", &var);
        v->Visit("val", &val);
    }
    bool SEqualReduce(const BindingNode* other, SEqualReducer equal) const {
        return equal(var, other->var) &&  equal(val, other->val);
    }
    void SHashReduce(SHashReducer hash_reduce) const {
        hash_reduce(var);
        hash_reduce(val);
    }
    static constexpr const char* _type_key = "relay2.expr.Binding";
    static constexpr const bool _type_has_method_sequal_reduce = true;
    static constexpr const bool _type_has_method_shash_reduce = true;
    TVM_DECLARE_FINAL_OBJECT_INFO(BindingNode, Object);
};

class Binding : public ObjectRef {
 public:
    TVM_DLL Binding(
        Var var,
        Expr val    );
    TVM_DEFINE_OBJECT_REF_METHODS(Binding, ObjectRef, BindingNode);
};

class Let;
class LetNode : public ExprNode {
 public:
    runtime::Array<Binding> bindings;
    Expr body;
    Span span;
    void VisitAttrs(AttrVisitor* v) {
        v->Visit("bindings", &bindings);
        v->Visit("body", &body);
        v->Visit("span", &span);
    }
    bool SEqualReduce(const LetNode* other, SEqualReducer equal) const {
        return equal(bindings, other->bindings) &&  equal(body, other->body) &&  equal(span, other->span);
    }
    void SHashReduce(SHashReducer hash_reduce) const {
        hash_reduce(bindings);
        hash_reduce(body);
        hash_reduce(span);
    }
    static constexpr const char* _type_key = "relay2.expr.Let";
    static constexpr const bool _type_has_method_sequal_reduce = true;
    static constexpr const bool _type_has_method_shash_reduce = true;
    TVM_DECLARE_FINAL_OBJECT_INFO(LetNode, ExprNode);
};

class Let : public Expr {
 public:
    TVM_DLL Let(
        runtime::Array<Binding> bindings,
        Expr body,
        Span span    );
    TVM_DEFINE_OBJECT_REF_METHODS(Let, Expr, LetNode);
};

class Function;
class FunctionNode : public ExprNode {
 public:
    runtime::String name;
    runtime::Array<Var> params;
    Expr body;
    Type ret_type;
    Span span;
    void VisitAttrs(AttrVisitor* v) {
        v->Visit("name", &name);
        v->Visit("params", &params);
        v->Visit("body", &body);
        v->Visit("ret_type", &ret_type);
        v->Visit("span", &span);
    }
    bool SEqualReduce(const FunctionNode* other, SEqualReducer equal) const {
        return equal(name, other->name) &&  equal(params, other->params) &&  equal(body, other->body) &&  equal(ret_type, other->ret_type) &&  equal(span, other->span);
    }
    void SHashReduce(SHashReducer hash_reduce) const {
        hash_reduce(name);
        hash_reduce(params);
        hash_reduce(body);
        hash_reduce(ret_type);
        hash_reduce(span);
    }
    static constexpr const char* _type_key = "relay2.expr.Function";
    static constexpr const bool _type_has_method_sequal_reduce = true;
    static constexpr const bool _type_has_method_shash_reduce = true;
    TVM_DECLARE_FINAL_OBJECT_INFO(FunctionNode, ExprNode);
};

class Function : public Expr {
 public:
    TVM_DLL Function(
        runtime::String name,
        runtime::Array<Var> params,
        Expr body,
        Type ret_type,
        Span span    );
    TVM_DEFINE_OBJECT_REF_METHODS(Function, Expr, FunctionNode);
};

class BroadcastShape;
class BroadcastShapeNode : public ExprNode {
 public:
    Expr lhs;
    Expr rhs;
    Span span;
    void VisitAttrs(AttrVisitor* v) {
        v->Visit("lhs", &lhs);
        v->Visit("rhs", &rhs);
        v->Visit("span", &span);
    }
    bool SEqualReduce(const BroadcastShapeNode* other, SEqualReducer equal) const {
        return equal(lhs, other->lhs) &&  equal(rhs, other->rhs) &&  equal(span, other->span);
    }
    void SHashReduce(SHashReducer hash_reduce) const {
        hash_reduce(lhs);
        hash_reduce(rhs);
        hash_reduce(span);
    }
    static constexpr const char* _type_key = "relay2.expr.BroadcastShape";
    static constexpr const bool _type_has_method_sequal_reduce = true;
    static constexpr const bool _type_has_method_shash_reduce = true;
    TVM_DECLARE_FINAL_OBJECT_INFO(BroadcastShapeNode, ExprNode);
};

class BroadcastShape : public Expr {
 public:
    TVM_DLL BroadcastShape(
        Expr lhs,
        Expr rhs,
        Span span    );
    TVM_DEFINE_OBJECT_REF_METHODS(BroadcastShape, Expr, BroadcastShapeNode);
};

class Dim;
class DimNode : public TypeNode {
 public:
    Span span;
    void VisitAttrs(AttrVisitor* v) {
        v->Visit("span", &span);
    }
    bool SEqualReduce(const DimNode* other, SEqualReducer equal) const {
        return equal(span, other->span);
    }
    void SHashReduce(SHashReducer hash_reduce) const {
        hash_reduce(span);
    }
    static constexpr const char* _type_key = "relay2.expr.Dim";
    static constexpr const bool _type_has_method_sequal_reduce = true;
    static constexpr const bool _type_has_method_shash_reduce = true;
    TVM_DECLARE_FINAL_OBJECT_INFO(DimNode, TypeNode);
};

class Dim : public Type {
 public:
    TVM_DLL Dim(
        Span span    );
    TVM_DEFINE_OBJECT_REF_METHODS(Dim, Type, DimNode);
};

class Shape;
class ShapeNode : public TypeNode {
 public:
    Span span;
    void VisitAttrs(AttrVisitor* v) {
        v->Visit("span", &span);
    }
    bool SEqualReduce(const ShapeNode* other, SEqualReducer equal) const {
        return equal(span, other->span);
    }
    void SHashReduce(SHashReducer hash_reduce) const {
        hash_reduce(span);
    }
    static constexpr const char* _type_key = "relay2.expr.Shape";
    static constexpr const bool _type_has_method_sequal_reduce = true;
    static constexpr const bool _type_has_method_shash_reduce = true;
    TVM_DECLARE_FINAL_OBJECT_INFO(ShapeNode, TypeNode);
};

class Shape : public Type {
 public:
    TVM_DLL Shape(
        Span span    );
    TVM_DEFINE_OBJECT_REF_METHODS(Shape, Type, ShapeNode);
};

class Tensor;
class TensorNode : public TypeNode {
 public:
    Optional<Expr> shape;
    Optional<Expr> dtype;
    Span span;
    void VisitAttrs(AttrVisitor* v) {
        v->Visit("shape", &shape);
        v->Visit("dtype", &dtype);
        v->Visit("span", &span);
    }
    bool SEqualReduce(const TensorNode* other, SEqualReducer equal) const {
        return equal(shape, other->shape) &&  equal(dtype, other->dtype) &&  equal(span, other->span);
    }
    void SHashReduce(SHashReducer hash_reduce) const {
        hash_reduce(shape);
        hash_reduce(dtype);
        hash_reduce(span);
    }
    static constexpr const char* _type_key = "relay2.expr.Tensor";
    static constexpr const bool _type_has_method_sequal_reduce = true;
    static constexpr const bool _type_has_method_shash_reduce = true;
    TVM_DECLARE_FINAL_OBJECT_INFO(TensorNode, TypeNode);
};

class Tensor : public Type {
 public:
    TVM_DLL Tensor(
        Optional<Expr> shape,
        Optional<Expr> dtype,
        Span span    );
    TVM_DEFINE_OBJECT_REF_METHODS(Tensor, Type, TensorNode);
};

} // namespace expr 
} // namespace relay2 
} // namespace tvm 
#endif  // TVM_RELAY2_EXPR_H_
