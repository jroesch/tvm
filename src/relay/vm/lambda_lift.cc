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

inline std::string GenerateName(const Function& func) {
    size_t hash = StructuralHash()(func);
    return std::string("lifted_name") + std::to_string(hash);
}

struct LambdaLifter : ExprMutator {
    Module module_;
    std::unordered_map<GlobalVar, Function, NodeHash, NodeEqual> lifted_;
    LambdaLifter(const Module& module) : module_(module) {}

    Expr VisitExpr_(const FunctionNode* func_node) final {
        auto func = GetRef<Function>(func_node);
        auto free_vars = FreeVars(func);
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

        body =
          FunctionSetAttr(body, "IsClosure", tvm::Integer(1));

        auto lifted_func =
            FunctionNode::make(
                args,
                body,
                func->func_type_annotation(),
                free_type_vars); // TODO(@jroesch), handle that

        lifted_func =
            FunctionSetAttr(lifted_func, "lifted", tvm::Integer(1));

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

    for (auto pair : module->functions) {
      auto global = pair.first;
      auto func = pair.second;
      updates.Set(global, lifter.Lift(func));
    }

    for (auto pair : lifter.lifted_) {
      updates.Set(pair.first, pair.second);
    }

    module->functions = updates;

    return module;
}

}  // namespace vm
}  // namespace relay
}  // namespace tvm
