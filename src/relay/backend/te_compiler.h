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

/*!
 * \file relay/backend/tir_compiler.h
 *  * \brief Internal compilation layer which lowers Relay "primitive functions" to TIR PrimFns.
 *
 *
 * This represents the new design of the Relay compilation flow and will replace the interface
 * contained in compile_engine.h as we migrate towards a standard pass based lowering of
 * Relay functions.
 *
 * This files provides an internal API which lowers Relay programs to components which
 * can be combined with TVM produced kernels to compile an entire program.
 *
 * The result of lowering contains a combination of `runtime::Module`s produced by external
 * compilers and a set of lowered PrimFns which can be code generated for targets.
 */
#ifndef TVM_RELAY_BACKEND_TE_COMPILER_H_
#define TVM_RELAY_BACKEND_TE_COMPILER_H_

#include <tvm/node/structural_equal.h>
#include <tvm/node/structural_hash.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/memory.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/op_strategy.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/module.h>
#include <tvm/topi/elemwise.h>

#include <functional>
#include <string>
#include <unordered_map>

#include "../transforms/infer_layout_utils.h"
#include "../transforms/pass_utils.h"
#include "./te_compiler_cache.h"

namespace tvm {
namespace relay {
namespace tec {

// TODO(@jroesch, @chrisS) these should be a tvm::Map for uniformity sake
// we should a version of context which works in Map
using TargetsMap = std::unordered_map<int, Target>;
using DeviceContextMap =
    std::unordered_map<Expr, TVMContext, runtime::ObjectPtrHash, runtime::ObjectPtrEqual>;

/*!
 * \brief A compiler which lowers primitive Relay functions to tensor expressions
 * and schdules them into TIR functions.
 */
class TECompilerNode : public Object {
 public:
  /*! \brief destructor */
  virtual ~TECompilerNode() {}
  /*!
   * \brief Get lowered result.
   * \param key The key to the cached function.
   * \return The result.
   */
  virtual CachedFunc Lower(const CCacheKey& key) = 0;

  virtual Map<String, IRModule> GetLoweredFunctions() = 0;
  /*!
   * \brief Just in time compile to get a PackedFunc.
   * \param key The key to the cached function.
   * \return The result.
   */
  virtual PackedFunc JIT(const CCacheKey& key) = 0;
  /*!
   * \brief Lower the shape function.
   * \param key The key to the cached function.
   * \return The result.
   */
  virtual CachedFunc LowerShapeFunc(const CCacheKey& key) = 0;
  /*!
   * \brief Lower the external function using external codegen tools.
   * \return The runtime moduels for each needed external codegen tool.
   */
  virtual tvm::Array<tvm::runtime::Module> LowerExternalFunctions() = 0;

  /*! \brief clear the cache. */
  virtual void Clear() = 0;

  // VisitAttrs
  void VisitAttrs(AttrVisitor*) {}

  static constexpr const char* _type_key = "relay.TECompiler";
  TVM_DECLARE_FINAL_OBJECT_INFO(TECompilerNode, Object);
};

/*! \brief cache entry used in compile engine */
class TECompiler : public ObjectRef {
 public:
  TECompiler();
  explicit TECompiler(ObjectPtr<Object> n) : ObjectRef(n) {}
  TECompilerNode* operator->() { return static_cast<TECompilerNode*>(get_mutable()); }
  using ContainerType = TECompilerNode;
  /*! \brief The global compile engine. */
  TVM_DLL static TECompiler& Global();
};

struct LoweredModule {
  IRModule main_module;
  Map<String, IRModule> per_target_module;
  Array<tvm::runtime::Module> external_mods;
};

LoweredModule LowerTE(const IRModule& module, TargetsMap targets,
                      DeviceContextMap device_context_map);

}  // namespace tec
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_BACKEND_TE_COMPILER_H_
