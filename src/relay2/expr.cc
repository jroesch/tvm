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
#include "/Users/jroesch/Git/tvm/include/relay2/expr.h"

namespace tvm { 
namespace relay2 { 
namespace expr { 

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
    relay::Id id) {
    ObjectPtr<VarNode> n = make_object<VarNode>();
    n->id = std::move(id);
    data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(VarNode);

TVM_REGISTER_GLOBAL("relay2.Var").set_body_typed([](relay::Id id) {
    return Var(id);
});

} // namespace expr 
} // namespace relay2 
} // namespace tvm 
