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
#include "/Users/jroesch/Git/tvm/include/relay2/expr.h"

namespace tvm { 
namespace relay2 { 
namespace expr { 

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
    static constexpr const char* _type_key = "expr.Expr";
    static constexpr const bool _type_has_method_sequal_reduce = true;
    static constexpr const bool _type_has_method_shash_reduce = true;
    TVM_DECLARE_BASE_OBJECT_INFO(ExprNode, Object);
};

class Expr : public ObjectRef {
 public:
    TVM_DLL Expr(
        Span span    );
    TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(Expr, ObjectRef, ExprNode);
};

class Var;
class VarNode : public Object {
 public:
    relay::Id id;
    void VisitAttrs(AttrVisitor* v) {
        v->Visit("id", &id);
    }
    bool SEqualReduce(const VarNode* other, SEqualReducer equal) const {
        return equal(id, other->id);
    }
    void SHashReduce(SHashReducer hash_reduce) const {
        hash_reduce(id);
    }
    static constexpr const char* _type_key = "expr.Var";
    static constexpr const bool _type_has_method_sequal_reduce = true;
    static constexpr const bool _type_has_method_shash_reduce = true;
    TVM_DECLARE_FINAL_OBJECT_INFO(VarNode, Object);
};

class Var : public ObjectRef {
 public:
    TVM_DLL Var(
        relay::Id id    );
    TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(Var, ObjectRef, VarNode);
};

} // namespace expr 
} // namespace relay2 
} // namespace tvm 
#endif  // TVM_RELAY2_EXPR_H_
