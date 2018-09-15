/*!
 *  Copyright (c) 2018 by Contributors
 * \file  environment.cc
 * \brief The global environment in Relay.
 */
#include <tvm/relay/environment.h>
#include <tvm/relay/pass.h>
#include <tvm/relay/pass/alpha_eq.h>
#include <sstream>
#include "./../pass/resolve.h"
// #include "tvm/relay/util/rang.h"

namespace tvm {
namespace relay {

using tvm::IRPrinter;
using namespace runtime;

Environment EnvironmentNode::make(tvm::Map<GlobalVar, Function> global_funcs) {
  std::shared_ptr<EnvironmentNode> n = std::make_shared<EnvironmentNode>();
  n->functions = std::move(global_funcs);
  return Environment(n);
}

GlobalVar EnvironmentNode::GetGlobalVar(const std::string &str) {
  auto global_id = global_map_.find(str);
  if (global_id != global_map_.end()) {
    return (*global_id).second;
  } else {
    auto id = GlobalVarNode::make(str);
    this->global_map_.Set(str, id);
    return id;
  }
}

/*! \brief Add a new item to the global environment
 * \note if the update flag is not set adding a duplicate
 * definition will trigger an exception, otherwise we will
 * update the definition if and only if it is type compatible.
 */
void EnvironmentNode::Add(const GlobalVar &var, const Function &func,
                          bool update) {
  // Type check the item before we add it to the environment.
  auto env = GetRef<Environment>(this);

  Expr checked_expr = InferType(env, var, func);

  if (const FunctionNode *func_node = checked_expr.as<FunctionNode>()) {
    auto checked_func = GetRef<Function>(func_node);
    auto type = checked_func->checked_type();

    CHECK(IsFullyResolved(type));

    if (functions.find(var) != functions.end()) {
      if (!update) {
        throw dmlc::Error("already have definition for XXXX.");
      }

      auto old_type = functions[var].as<FunctionNode>()->checked_type();

      if (!AlphaEqual(type, old_type)) {
        throw dmlc::Error(
            "Environment#update changes type, not possible in this mode.");
      }

      this->functions.Set(var, checked_func);
    } else {
      this->functions.Set(var, checked_func);
    }
  } else {
    throw Error("internal error: unknown item type, unreachable code");
  }
}

void EnvironmentNode::Update(const GlobalVar &var, const Function &func) {
  this->Add(var, func, true);
}

void EnvironmentNode::Remove(const GlobalVar &) {
  // Clarify with @tqchen about how to use COW to do this.
  throw Error("NYI");
  // this->items.erase(id);
}

Function EnvironmentNode::Lookup(const GlobalVar &var) {
  auto func = functions.find(var);
  if (func != functions.end()) {
    return (*func).second;
  } else {
    throw Error(std::string("there is no definition of ") + var->name_hint);
  }
}

Function EnvironmentNode::Lookup(const std::string &str) {
  GlobalVar id = this->GetGlobalVar(str);
  return this->Lookup(id);
}

void EnvironmentNode::Merge(const Environment &env) {
  for (auto pair : env->functions) {
    this->functions.Set(pair.first, pair.second);
  }
}

void EnvironmentNode::Transform(EnvironmentNode::Transformer transformer) {
  Array<GlobalVar> to_process;
  for (auto var_and_func : this->functions) {
    to_process.push_back(var_and_func.first);
  }

  auto for_each = transformer(GetRef<Environment>(this));
  for (auto var : to_process) {
    auto func = this->functions[var];
    auto transformed = for_each(var, func);
    this->Add(var, transformed, true);
  }
}

TVM_REGISTER_API("relay._make.Environment")
    .set_body([](TVMArgs args, TVMRetValue *ret) {
      *ret = EnvironmentNode::make(args[0]);
    });

TVM_REGISTER_API("relay._env.Environment_Add")
    .set_body([](TVMArgs args, TVMRetValue *ret) {
      Environment env = args[0];
      env->Add(args[1], args[2], false);
    });

TVM_REGISTER_API("relay._env.Environment_GetGlobalVar")
    .set_body([](TVMArgs args, TVMRetValue *ret) {
      Environment env = args[0];
      *ret = env->GetGlobalVar(args[1]);
    });

TVM_REGISTER_API("relay._env.Environment_Lookup")
    .set_body([](TVMArgs args, TVMRetValue *ret) {
      Environment env = args[0];
      GlobalVar var = args[1];
      *ret = env->Lookup(var);
    });

TVM_REGISTER_API("relay._env.Environment_Lookup_str")
    .set_body([](TVMArgs args, TVMRetValue *ret) {
      Environment env = args[0];
      std::string var_name = args[1];
      auto var = env->GetGlobalVar(var_name);
      *ret = env->Lookup(var);
    });

TVM_REGISTER_API("relay._env.Environment_Merge")
    .set_body([](TVMArgs args, TVMRetValue *ret) {
      Environment env = args[0];
      env->Merge(args[1]);
    });

TVM_REGISTER_API("relay._env.Environment_Transform")
    .set_body([](TVMArgs args, TVMRetValue *ret) {
      Environment env = args[0];
      env->Transform(args[1]);
    });

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
    .set_dispatch<EnvironmentNode>([](const EnvironmentNode *node,
                                      tvm::IRPrinter *p) {
      p->stream << "EnvironmentNode( " << node->functions << ")";
    });

}  // namespace relay
}  // namespace tvm
