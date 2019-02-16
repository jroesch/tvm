/*!
 *  Copyright (c) 2018 by Contributors
 * \file tvm/relay/vm/lambda_lift.cc
 * \brief Lift all nested functions into global functions.
 */

#include <tvm/runtime/memory_manager.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/vm/vm.h>
#include <tvm/relay/interpreter.h>
#include "../backend/compile_engine.h"
#include "../../runtime/naive_allocator.h"

#include <vector>
#include <iostream>

using namespace tvm::runtime;

namespace tvm {
namespace relay {
namespace vm {

static const char* kIsClosure = "IsClosure";

inline std::string GenerateName(const Function& func) {
  size_t hash = StructuralHash()(func);
  return std::string("lifted_name") + std::to_string(hash);
}

bool IsClosure(const Function& func) {
  NodeRef res = FunctionGetAttr(func, kIsClosure);
  const ir::IntImm* pval = res.as<ir::IntImm>();
  return pval && pval->value != 0;
}

Function MarkClosure(const Function& func) {
  return FunctionSetAttr(func, kIsClosure, tvm::Integer(1));
}

struct LambdaLifter : ExprMutator {
    Module module_;
    std::unordered_map<GlobalVar, Function, NodeHash, NodeEqual> lifted_;
    LambdaLifter(const Module& module) : module_(module) {}

    Expr VisitExpr_(const FunctionNode* func_node) final {
        auto func = GetRef<Function>(func_node);

        // We should not transform primitive functions.
        if (func->IsPrimitive()) {
          return func;
        }

        auto free_vars = FreeVars(func);

        // If there are no free variables this transform is easy.
        //
        // We maybe should eta-global an lift?
        if (free_vars.size() == 0) {
          auto name = GenerateName(func);
          auto global = this->module_->GetGlobalVar(name);
          auto vfunc = Downcast<Function>(ExprMutator::VisitExpr_(func_node));
          lifted_.insert({global, vfunc });
          return global;
        }

        auto free_type_vars = FreeTypeVars(func, module_);

        tvm::Map<Var, Expr> subst_map;
        tvm::Array<Var> args;
        size_t i = 0;
        for (auto fv : free_vars) {
            std::string name = "free_var" + std::to_string(i++);
            auto arg = VarNode::make(name, fv->type_annotation);
            subst_map.Set(fv, arg);
            args.push_back(arg);
        }

        auto body = Downcast<Function>(Bind(func, subst_map));

        auto lifted_func =
            FunctionNode::make(
                args,
                body,
                func->func_type_annotation(),
                free_type_vars); // TODO(@jroesch), handle that

        lifted_func = MarkClosure(lifted_func);

        auto name = GenerateName(lifted_func);
        auto global = this->module_->GetGlobalVar(name);
        lifted_.insert({global, lifted_func});

        // Finally we bind the variables here to
        // explicitly capture the closure.
        Array<Expr> fvs;
        for (auto fv : free_vars ) { fvs.push_back(fv); }

        return CallNode::make(global, fvs);
    }

    Function Lift(const Function& func) {
        return FunctionNode::make(
            func->params,
            VisitExpr(func->body),
            func->ret_type,
            func->type_params,
            func->attrs);
    }
};

/* The goal of this pass is to lift out any nested functions into top-level
 * functions.
 *
 * We will lift the functions out into globals which take the set of the free vars
 * and then return a function whcih has b
 */
Module LambdaLift(const Module& module)  {
    LambdaLifter lifter(module);

    tvm::Map<GlobalVar, Function> updates;

    // There is an ordering bug here.
    for (auto pair : module->functions) {
      auto global = pair.first;
      auto func = pair.second;
      updates.Set(global, lifter.Lift(func));
    }

    for (auto pair : lifter.lifted_) {
      module->Add(pair.first, pair.second);
    }

    for (auto pair : updates) {
      module->Add(pair.first, pair.second, true);
    }

    return module;
}

}  // namespace vm
}  // namespace relay
}  // namespace tvm
