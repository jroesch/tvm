/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include "te_compiler.h"

#include <tvm/driver/driver_api.h>
#include <tvm/ir/type_functor.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/device_copy.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/runtime/container.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>
#include <tvm/te/operation.h>
#include <tvm/te/schedule.h>
#include <tvm/te/schedule_pass.h>
#include <tvm/topi/tags.h>

#include <functional>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../transforms/pass_utils.h"
#include "te_compiler_cache.h"
#include "utils.h"

namespace tvm {
namespace relay {
namespace tec {

using namespace tvm::relay::transform;

TVM_REGISTER_OBJECT_TYPE(TECompilerNode);

class TECompilerImpl : public TECompilerNode {
 public:
  // Lower the function.
  CachedFunc Lower(const CCacheKey& key) { return LowerInternal(key)->cached_func; }

  // For now, build one module per function.
  PackedFunc JIT(const CCacheKey& key) final {
    CCacheValue value = LowerInternal(key);
    if (value->packed_func != nullptr) {
      return value->packed_func;
    }
    auto m = build(value->cached_func->funcs, key->target, Target(nullptr));
    value->packed_func = m.GetFunction(value->cached_func->prim_fn_var->name_hint);
    return value->packed_func;
  }

  CachedFunc LowerShapeFunc(const CCacheKey& key) final {
    return LowerShapeFuncInternal(key)->cached_func;
  }

  Map<String, IRModule> GetLoweredFunctions() {
    Map<String, IRModule> lowered_functions;
    for (const auto& it : cache_) {
      auto source_func = it.first;
      auto lowered_func = it.second;
      auto target = source_func->target;

      if (!lowered_functions.count(target->str())) {
        lowered_functions.Set(target->str(), IRModule(Map<GlobalVar, BaseFunc>({})));
      }

      lowered_functions[target->str()]->Update(lowered_func->cached_func->funcs);
    }
    return lowered_functions;
  }

  Array<tvm::runtime::Module> LowerExternalFunctions() {
    Array<tvm::runtime::Module> ret;
    std::unordered_map<std::string, std::string> cached_symbol;
    std::vector<CCacheKey> cached_ext_funcs;
    for (const auto& it : cache_) {
      auto src_func = it.first->source_func;
      ICHECK(src_func.defined());
      if (src_func->GetAttr<String>(attr::kCompiler).defined()) {
        auto code_gen = src_func->GetAttr<String>(attr::kCompiler);
        std::string code_gen_name = code_gen.value();
        cached_ext_funcs.push_back(it.first);

        auto symbol_name = src_func->GetAttr<String>(tvm::attr::kGlobalSymbol);
        ICHECK(symbol_name.defined()) << "No external symbol is set for:\n"
                                      << AsText(src_func, false);

        std::string sn = symbol_name.value();
        if (cached_symbol.count(sn)) {
          cached_symbol[sn] = code_gen_name;
        } else {
          ICHECK_NE(sn, code_gen_name)
              << "Found duplicated symbol: " << sn << " for: " << code_gen_name;
        }

        std::string ext_name = "relay.ext." + code_gen_name;
        auto pf = tvm::runtime::Registry::Get(ext_name);
        ICHECK(pf) << "Failed to find the codegen tool for " << ext_name;
        // No need to keep compiler attribute at this point, functions have been
        // extracted for specific codegen.
        src_func = WithAttr(std::move(src_func), attr::kCompiler, NullValue<ObjectRef>());
        runtime::Module ext_mod = (*pf)(src_func);

        ICHECK(ext_mod.defined()) << "No external runtime is generated.";
        ret.push_back(ext_mod);
      }
    }

    // No need to cache external functions as we collected them all to create
    // external runtime modules.
    for (const auto& it : cached_ext_funcs) {
      cache_.erase(it);
    }
    return ret;
  }

  void Clear() final { cache_.clear(); }

  // List all items in the cache.
  Array<ObjectRef> ListItems() {
    std::lock_guard<std::mutex> lock(mutex_);
    Array<ObjectRef> items;
    for (auto& kv : cache_) {
      items.push_back(kv.first);
      items.push_back(kv.second);
    }
    return items;
  }

  /*!
   * \brief Get the cache key of the function that is being lowered currently
   * \return the cache key
   */
  CCacheKey GetCurrentCCacheKey() { return cur_ccache_key_; }

 private:
  // implement lowered func
  CCacheValue LowerInternal(const CCacheKey& key) {
    std::lock_guard<std::mutex> lock(mutex_);
    CCacheValue value;
    auto it = cache_.find(key);
    if (it != cache_.end()) {
      it->second->use_count += 1;
      if (it->second->cached_func.defined()) return it->second;
      value = it->second;
    } else {
      value = CCacheValue(make_object<CCacheValueNode>());
      value->use_count = 0;
      if (!backend::IsCompileEngineCacheDisabled()) {
        cache_[key] = value;
      }
    }
    cur_ccache_key_ = key;

    // No need to lower external functions for now. We will invoke the external
    // codegen tool once and lower all functions together.
    if (key->source_func->GetAttr<String>(attr::kCompiler).defined()) {
      auto ir_module = IRModule();
      const auto name_node = key->source_func->GetAttr<String>(tvm::attr::kGlobalSymbol);
      ICHECK(name_node.defined()) << "External function has not been attached a name yet.";
      auto func_name = GetUniqueName(name_node.value(), &name_map_);
      auto target = Target("ext_dev");
      auto global_var = GlobalVar(func_name);
      global_var->checked_type_ = key->source_func->checked_type();
      value->cached_func = CachedFunc(target, global_var, {}, {}, te::Schedule(), {}, ir_module);
      return value;
    }
    // Enforce use the target.
    With<Target> target_scope(key->target);

    ICHECK(!value->cached_func.defined());
    auto cfunc = PrimFuncFor(key->source_func, key->target,
                             [&](std::string name) { return GetUniqueName(name, &name_map_); });

    // Skip lowering for device copy node.
    const Expr body = (key->source_func)->body;
    if (const CallNode* call_node = body.as<CallNode>()) {
      if (call_node->attrs.as<DeviceCopyAttrs>()) {
        value->cached_func = cfunc;
        return value;
      }
    }

    // NOTE: array will copy on write.
    Array<te::Tensor> all_args = Array<te::Tensor>(cfunc->inputs);
    for (te::Tensor arg : cfunc->outputs) {
      all_args.push_back(arg);
    }

    std::unordered_map<te::Tensor, tir::Buffer> binds;
    auto func_name = cfunc->prim_fn_var->name_hint;
    cfunc->funcs->Update(TVMLower(cfunc->schedule, all_args, func_name, key->source_func, binds));
    value->cached_func = cfunc;
    return value;
  }

  // implement lowered shape func
  CCacheValue LowerShapeFuncInternal(const CCacheKey& key) {
    std::lock_guard<std::mutex> lock(mutex_);
    CCacheValue value;
    auto it = shape_func_cache_.find(key);
    if (it != shape_func_cache_.end()) {
      it->second->use_count += 1;
      if (it->second->cached_func.defined()) return it->second;
      value = it->second;
    } else {
      value = CCacheValue(make_object<CCacheValueNode>());
      value->use_count = 0;
      shape_func_cache_[key] = value;
    }
    // Enforce use the target.
    With<Target> target_scope(key->target);

    ICHECK(!value->cached_func.defined());

    using tvm::transform::PassContext;
    With<PassContext> fresh_pass_ctx_scope(PassContext::Create());
    auto cached_func = ShapeFuncFor(key->source_func, key->target, [&](std::string name) {
      return GetUniqueName(name, &name_map_);
    });

    value->cached_func = cached_func;
    return value;
  }

  /*! \brief compiler cache lock*/
  std::mutex mutex_;
  /*! \brief internal name map to get an unique name */
  std::unordered_map<std::string, int> name_map_;
  /*! \brief internal compiler cache */
  std::unordered_map<CCacheKey, CCacheValue> cache_;
  /*! \brief internal compiler cache for shape funcs */
  std::unordered_map<CCacheKey, CCacheValue> shape_func_cache_;
  /*! \brief the cache key of the function that is being lowered currently*/
  CCacheKey cur_ccache_key_;
};

TECompiler::TECompiler() {
  auto object = make_object<TECompilerImpl>();
  data_ = object;
}

class LowerTensorExpr : public ExprMutator {
 public:
  LowerTensorExpr(const IRModule& module, const TargetMap& targets, const DeviceMap& device_ctx_map,
                  ProcessFn process_fn, TECompiler compiler)
      : module_(module),
        targets_(targets),
        device_context_map_(device_ctx_map),
        process_fn(process_fn),
        compiler_(compiler) {}

  // bool ShareSameStorage(const Expr& lhs, const Expr& rhs) {
  //   auto lit = storage_device_map_.find(lhs);
  //   auto rit = storage_device_map_.find(rhs);
  //   ICHECK(lit != storage_device_map_.end());
  //   ICHECK(rit != storage_device_map_.end());
  //   int64_t lhs_storage_id = ((*lit).second)[0][0]->value;
  //   int64_t rhs_storage_id = ((*rit).second)[0][0]->value;
  //   return lhs_storage_id == rhs_storage_id;
  // }

  Expr VisitExpr_(const CallNode* call) override {
    Call expr = GetRef<Call>(call);
    Function func;

    if (call->op.as<FunctionNode>()) {
      func = GetRef<Function>(call->op.as<FunctionNode>());
    } else {
      return ExprMutator::VisitExpr_(call);
    }

    // Provide a callback hook which allows one-level up code generators to
    // act when we process a function.
    this->process_fn(func);

    if (!func->HasNonzeroAttr(attr::kPrimitive)) {
      return ExprMutator::VisitExpr_(call);
    }

    // Process inputs.
    Array<Expr> args;
    for (size_t i = 0; i < expr->args.size(); i++) {
      args.push_back(VisitExpr(expr->args[i]));
    }

    Target target;

    if (func->GetAttr<String>(attr::kCompiler).defined()) {
      target = Target("ext_dev");
      CCacheKey key = CCacheKey(func, target);
      CachedFunc ext_func = compiler_->Lower(key);
      ICHECK(ext_func.defined()) << "Lowering returned undefined function for "
                                 << ext_func->prim_fn_var->name_hint;
      return Call(ext_func->prim_fn_var, args, {});
    }

    ICHECK_GE(device_context_map_.count(expr), 0)
          << "Could not find an entry in the device context map for " << PrettyPrint(expr)
          << "The memory planning was either not performed for this precise node, or there is bug "
             "in the memory planner.";

    auto& device_context = this->device_context_map_[expr];
    auto call_dev_type = device_context.device_type;

    // Non-External Relay Function
    if (targets_.size() == 1) {
      // The homogeneous execution case, we should only have one target
      // so we just grab it.
      const auto& it = targets_.begin();
      target = (*it).second;
    } else {
      std::cout << "DeviceType: " << call_dev_type << std::endl;
      // The heterogeneous execution case we have multiple targets
      // in this case.
      //
      // We need to identify the target and translate.
      std::string call_dev_name;
      if (call_dev_type == 0) {
        call_dev_name = "llvm";
        call_dev_type = kDLCPU;
      } else {
        call_dev_name = ::tvm::runtime::DeviceName(call_dev_type);
      }

      if (targets_.count(call_dev_type) == 0) {
          std::stringstream msg;
          msg << "No target is specified for provided device name: `" << call_dev_name << "`\n\n";
          msg << call_dev_name << " mapped to device type (" << call_dev_type << ") which was not found in the target map.\n";
          msg << "Availible targets: \n";
          for (auto target : targets_) {
            msg << "  " << target.first << "-> " << target.second << "\n";
          }
          LOG(FATAL) << msg.str();
      }

      std::cout << "DeviceName: " << call_dev_name << std::endl;
      target = targets_[call_dev_type];
      std::cout << "Target: " << target << std::endl;
    }

    CCacheKey key = CCacheKey(func, target);
    CachedFunc lowered_func = compiler_->Lower(key);

    return Call(lowered_func->prim_fn_var, args, Attrs());
  }

  IRModule module_;
  TargetMap targets_;
  DeviceMap device_context_map_;
  ProcessFn process_fn;
  TECompiler compiler_;
};

LoweredModule LowerTE(const IRModule& module, TargetMap targets, DeviceMap device_context_map,
                      std::function<void(Function)> process_fn) {
  TECompiler compiler;

  auto pass = CreateFunctionPass(
      [=](Function func, IRModule module, PassContext ctx) {
        LowerTensorExpr lower_te(module, targets, device_context_map, process_fn, compiler);
        return Downcast<Function>(lower_te.VisitExpr(func));
      },
      0, "LowerTensorExpr", {});

  auto updated_module = pass(module);

  LoweredModule lowered_module;
  lowered_module.main_module = updated_module;
  lowered_module.per_target_module = compiler->GetLoweredFunctions();
  lowered_module.external_mods = compiler->LowerExternalFunctions();
  return lowered_module;
}

}  // namespace tec
}  // namespace relay
}  // namespace tvm
