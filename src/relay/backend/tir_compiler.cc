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
 * \file relay/backend/tir_compiler.cc
 * \brief An internal API for lowering the TIR portion of the
 * module to runtime::Module.
 */
#include "tir_compiler.h"

#include <tvm/driver/driver_api.h>
#include <tvm/ir/type_functor.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/device_copy.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/runtime/container.h>
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
#include "utils.h"

namespace tvm {
namespace relay {
namespace tirc {

using namespace tvm::relay::transform;

TVM_REGISTER_NODE_TYPE(LoweredOutputNode);
TVM_REGISTER_NODE_TYPE(CachedFuncNode);
TVM_REGISTER_NODE_TYPE(CCacheKeyNode);
TVM_REGISTER_NODE_TYPE(CCacheValueNode);
TVM_REGISTER_OBJECT_TYPE(TECompilerNode);

LoweredOutput::LoweredOutput(tvm::Array<te::Tensor> outputs, OpImplementation impl) {
  auto n = make_object<LoweredOutputNode>();
  n->outputs = std::move(outputs);
  n->implementation = std::move(impl);
  data_ = std::move(n);
}

CCacheKey::CCacheKey(Function source_func, Target target) {
  auto n = make_object<CCacheKeyNode>();
  n->source_func = std::move(source_func);
  n->target = std::move(target);
  data_ = std::move(n);
}

Array<IndexExpr> GetShape(const Array<IndexExpr>& shape) {
  // for now, we always use int32 shape when possible
  // even if the result of shape inference becomes int64.
  Array<IndexExpr> res;
  for (IndexExpr val : shape) {
    const int64_t* pval = tir::as_const_int(val);
    if (pval != nullptr) {
#ifndef TVM_INDEX_DEFAULT_I64
      ICHECK_LE(pval[0], std::numeric_limits<int32_t>::max());
      ICHECK_GE(pval[0], std::numeric_limits<int32_t>::min());
      res.push_back(IntImm(DataType::Int(32), *pval));
#else
      res.push_back(val);
#endif  // TVM_INDEX_DEFAULT_I64
    } else if (val->IsInstance<tir::AnyNode>()) {
      res.push_back(val.as<tir::AnyNode>()->ToVar());
    } else {
      res.push_back(val);
    }
  }
  return res;
}

// The getter to get schedule from compile engine.
// Get schedule from functor.
class ScheduleGetter : public backend::MemoizedExprTranslator<Array<te::Tensor>> {
 public:
  explicit ScheduleGetter(Target target)
      : target_(target), device_copy_op_(Op::Get("device_copy")) {
    // Whether to use auto_scheduler schedule.
    use_auto_scheduler_ = backend::IsAutoSchedulerEnabled();
  }

  CachedFunc Create(const Function& prim_func) {
    auto cache_node = make_object<CachedFuncNode>();
    cache_node->target = target_;
    for (Var param : prim_func->params) {
      Array<tvm::te::Tensor> inputs;
      if (const auto* ttype = param->checked_type().as<TensorTypeNode>()) {
        tvm::te::Tensor tensor = tvm::te::placeholder(GetShape(ttype->shape), ttype->dtype);
        cache_node->inputs.push_back(tensor);
        inputs.push_back(tensor);
      } else {
        // flatten tuple of tensor type.
        const auto* tuple_type = param->type_as<TupleTypeNode>();
        for (Type field : tuple_type->fields) {
          const auto* ttype = field.as<TensorTypeNode>();
          // TODO(@icemelon): Allow recursive tuple
          ICHECK(ttype != nullptr);
          tvm::te::Tensor tensor = tvm::te::placeholder(GetShape(ttype->shape), ttype->dtype);
          cache_node->inputs.push_back(tensor);
          inputs.push_back(tensor);
        }
      }
      memo_[param] = inputs;
    }
    readable_name_stream_ << "fused";
    cache_node->outputs = this->VisitExpr(prim_func->body);
    auto candidate_name = readable_name_stream_.str();
    constexpr static size_t kMaxFuncNameLength = 80;
    if (candidate_name.size() > kMaxFuncNameLength) {
      std::stringstream truncated_name;
      truncated_name << candidate_name.substr(0, kMaxFuncNameLength);
      truncated_name << "_" << std::hash<std::string>{}(candidate_name) << "_";
      candidate_name = truncated_name.str();
    }
    cache_node->func_name = candidate_name;
    ICHECK(anchor_op_.defined());
    // Fusion over tupled results may leave identity relationships
    // between inputs and outputs, and those should not be scheduled.
    // Hence schedule only non PlaceholderOp outputs.
    tvm::Array<te::Tensor> tensor_outs;
    for (const auto& tensor : cache_node->outputs) {
      if (!tensor->op.as<te::PlaceholderOpNode>()) {
        tensor_outs.push_back(tensor);
      }
    }

    te::Schedule schedule;
    // No need to register schedule for device copy op.
    if (anchor_attrs_.as<DeviceCopyAttrs>() == nullptr) {
      if (use_auto_scheduler_) {
        const auto* fauto_schedule =
            runtime::Registry::Get("auto_scheduler.relay_integration.auto_schedule_topi_compute");
        ICHECK(fauto_schedule != nullptr)
            << "auto_scheduler.relay_integration.auto_schedule_topi_compute is not registered";
        ObjectRef obj = (*fauto_schedule)(tensor_outs);
        if (obj.defined()) {
          schedule = Downcast<te::Schedule>(obj);
        }
      }

      // Use TOPI schdule if user specificed, or the function has no auto_scheduler schedule.
      if (!schedule.defined()) {
        ICHECK(anchor_implementation_.defined());
        schedule = anchor_implementation_.Schedule(anchor_attrs_, tensor_outs, target_);
      }
      for (const auto& scalar : scalars_) {
        if (schedule->Contain(scalar)) {
          schedule[scalar].compute_inline();
        }
      }
    }
    cache_node->schedule = std::move(schedule);
    return CachedFunc(cache_node);
  }

  Array<te::Tensor> VisitExpr_(const VarNode* op) final {
    LOG(FATAL) << "Free variable " << op->name_hint();
    return {};
  }

  Array<te::Tensor> VisitExpr_(const ConstantNode* op) final {
    using tir::make_const;
    ICHECK(op->is_scalar());
    void* data = op->data->data;
    DataType dtype = DataType(op->data->dtype);
    auto value = te::compute(
        {},
        [&](const Array<tvm::tir::Var>&) {
          if (dtype == DataType::Int(32)) {
            return make_const(dtype, static_cast<const int32_t*>(data)[0]);
          } else if (dtype == DataType::Int(64)) {
            return make_const(dtype, static_cast<const int64_t*>(data)[0]);
          } else if (dtype == DataType::Float(32)) {
            return make_const(dtype, static_cast<const float*>(data)[0]);
          } else if (dtype == DataType::Float(64)) {
            return make_const(dtype, static_cast<const double*>(data)[0]);
          } else if (dtype == DataType::Bool()) {
            return make_const(dtype, static_cast<const uint8_t*>(data)[0]);
          } else {
            LOG(FATAL) << "not handled";
            return tvm::PrimExpr();
          }
        },
        "compile_engine_const", topi::kBroadcast);
    scalars_.push_back(value->op);
    return {value};
  }

  Array<te::Tensor> VisitExpr_(const CallNode* call_node) final {
    static auto fpattern = Op::GetAttrMap<TOpPattern>("TOpPattern");
    static auto flower_call = tvm::runtime::Registry::Get("relay.backend.lower_call");
    ICHECK(flower_call) << "relay.backend.lower_call is not registered.";

    Array<te::Tensor> inputs;
    int count_tuple = 0;
    for (Expr arg : call_node->args) {
      if (arg->checked_type().as<TupleTypeNode>()) {
        ++count_tuple;
      }
      for (te::Tensor tensor : VisitExpr(arg)) {
        inputs.push_back(tensor);
      }
    }
    if (count_tuple) {
      ICHECK_EQ(call_node->args.size(), 1U) << "Only allow function with a single tuple input";
    }

    ICHECK(call_node->op.as<OpNode>()) << "Primitive function only allows call into primitive ops";
    Op op = Downcast<Op>(call_node->op);

    Array<te::Tensor> outputs;
    OpImplementation impl;
    // Skip fcompute for device copy operators as it is not registered.
    if (op == device_copy_op_) {
      const auto* copy_input = inputs[0].operator->();
      outputs.push_back(te::Tensor(copy_input->shape, copy_input->dtype, te::Operation(), 0));
    } else {
      LoweredOutput lowered_out = (*flower_call)(GetRef<Call>(call_node), inputs, target_);
      outputs = lowered_out->outputs;
      impl = lowered_out->implementation;
    }

    int op_pattern = fpattern[op];
    if (!use_auto_scheduler_ && op_pattern >= kCommReduce) {
      ICHECK(!anchor_op_.defined() || anchor_op_pattern_ < kCommReduce)
          << "Cannot apply TOPI schedule to a primitive function with two complicated ops"
          << " anchor=" << anchor_op_ << " current=" << op;
    }
    if (op_pattern >= anchor_op_pattern_) {
      anchor_op_ = op;
      anchor_attrs_ = call_node->attrs;
      anchor_op_pattern_ = op_pattern;
      anchor_implementation_ = impl;
    }
    if (outputs.size() != 1) {
      const auto* tuple_type = call_node->checked_type().as<TupleTypeNode>();
      ICHECK(tuple_type) << "Expect output to be a tuple type";
      ICHECK_EQ(tuple_type->fields.size(), outputs.size());
    }
    // Set the name to `__copy`. It will be detected in graph runtime to perform
    // data copy across devices.
    if (op == device_copy_op_) {
      readable_name_stream_.str(std::string());
      readable_name_stream_ << "__copy";
    } else {
      readable_name_stream_ << '_' << op->name;
    }
    return outputs;
  }

  Array<te::Tensor> VisitExpr_(const FunctionNode* op) final {
    LOG(FATAL) << "Do not support sub function";
    return Array<te::Tensor>();
  }

  Array<te::Tensor> VisitExpr_(const LetNode* op) final {
    Array<te::Tensor> val = VisitExpr(op->value);
    ICHECK(!memo_.count(op->var));
    memo_[op->var] = val;
    return VisitExpr(op->body);
  }

  Array<te::Tensor> VisitExpr_(const TupleNode* op) final {
    Array<te::Tensor> fields;
    for (Expr field : op->fields) {
      ICHECK(field->checked_type().as<TensorTypeNode>()) << "Only allow Tuple of Tensor";
      Array<te::Tensor> res = VisitExpr(field);
      ICHECK_EQ(res.size(), 1);
      fields.push_back(res[0]);
    }
    return fields;
  }

  Array<te::Tensor> VisitExpr_(const TupleGetItemNode* op) final {
    const auto* tuple_type = op->tuple->type_as<TupleTypeNode>();
    Array<te::Tensor> tuple = VisitExpr(op->tuple);
    ICHECK_EQ(tuple_type->fields.size(), tuple.size());
    ICHECK_GE(op->index, 0);
    ICHECK_LT(static_cast<size_t>(op->index), tuple.size());
    return {tuple[op->index]};
  }

 private:
  tvm::Target target_;
  Op anchor_op_;
  Attrs anchor_attrs_;
  int anchor_op_pattern_{0};
  OpImplementation anchor_implementation_;
  std::ostringstream readable_name_stream_;
  Array<te::Operation> scalars_;
  bool use_auto_scheduler_;
  // Cache device copy op for equivalence checking to reduce registry lookup
  // overhead for each invocation of call node when retrieving schedules.
  const Op& device_copy_op_;
};

/*!
 * \brief Create schedule for target.
 * \param source_func The primitive function to be lowered.
 * \param target The target we want to create schedule for.
 * \return Pair of schedule and cache.
 *  The funcs field in cache is not yet populated.
 */
CachedFunc CreateSchedule(const Function& source_func, const Target& target) {
  return ScheduleGetter(target).Create(source_func);
}

// Creates shape function from functor.
class MakeShapeFunc : public backend::MemoizedExprTranslator<Array<te::Tensor>> {
 public:
  MakeShapeFunc() {}

  std::pair<te::Schedule, CachedFunc> Create(const Function& prim_func) {
    for (auto param : prim_func->params) {
      param_states_[param] = kNoNeed;
      Array<tvm::te::Tensor> data_inputs;
      Array<tvm::te::Tensor> shape_inputs;

      auto add_placeholder = [&data_inputs, &shape_inputs](const TensorTypeNode* ttype) {
        // Add data placeholder
        Shape shape = GetShape(ttype->shape);
        tvm::te::Tensor data_tensor = tvm::te::placeholder(shape, ttype->dtype);
        data_inputs.push_back(data_tensor);
        // Add shape placeholder
        int64_t ndim = shape.size();
        Shape sshape;
        if (ndim > 0) {
          sshape.push_back(tvm::Integer(ndim));
        }
        tvm::te::Tensor shape_tensor = tvm::te::placeholder(sshape, DataType::Int(64));
        shape_inputs.push_back(shape_tensor);
      };

      if (const auto* ttype = param->checked_type().as<TensorTypeNode>()) {
        add_placeholder(ttype);
      } else {
        // flatten tuple of tensor type.
        const auto* tuple_type = param->type_as<TupleTypeNode>();
        // TODO(@icemelon): Support recursive tuple
        ICHECK(tuple_type);
        for (Type field : tuple_type->fields) {
          const auto* ttype = field.as<TensorTypeNode>();
          ICHECK(ttype);
          add_placeholder(ttype);
        }
      }
      param_data_[param] = data_inputs;
      param_shapes_[param] = shape_inputs;
    }
    readable_name_stream_ << "shape_func";
    auto cache_node = make_object<CachedFuncNode>();
    cache_node->outputs = VisitExpr(prim_func->body);
    auto candidate_name = readable_name_stream_.str();
    constexpr static size_t kMaxFuncNameLength = 80;
    if (candidate_name.size() > kMaxFuncNameLength) {
      std::stringstream truncated_name;
      truncated_name << candidate_name.substr(0, kMaxFuncNameLength);
      truncated_name << "_" << std::hash<std::string>{}(candidate_name) << "_";
      candidate_name = truncated_name.str();
    }
    cache_node->func_name = candidate_name;

    // set inputs
    for (auto param : prim_func->params) {
      int state = param_states_[param];
      cache_node->shape_func_param_states.push_back(IntImm(DataType::Int(32), state));
      if (state & kNeedInputData) {
        for (auto t : param_data_[param]) {
          cache_node->inputs.push_back(t);
        }
      }
      if (state & kNeedInputShape) {
        for (auto t : param_shapes_[param]) {
          cache_node->inputs.push_back(t);
        }
      }
    }

    CachedFunc cfunc(cache_node);
    // generate schedule for shape func
    Array<te::Operation> out_ops;
    for (auto t : cache_node->outputs) {
      out_ops.push_back(t->op);
    }
    auto schedule = te::create_schedule(out_ops);
    tvm::te::AutoInlineInjective(schedule);
    for (const auto& scalar : scalars_) {
      auto scalar_op = scalar->op;
      if (schedule->Contain(scalar_op)) {
        schedule[scalar_op].compute_inline();
      }
    }
    return std::make_pair(schedule, cfunc);
  }

  Array<te::Tensor> VisitExpr(const Expr& expr) final {
    if (expr.as<VarNode>()) {
      // Do not memoize vars because shape functions could use either the data
      // or the shape of a var each time.
      return ExprFunctor::VisitExpr(expr);
    }
    // For other case, do memoized visit
    return backend::MemoizedExprTranslator<Array<te::Tensor>>::VisitExpr(expr);
  }

  Array<te::Tensor> VisitExpr_(const VarNode* var_node) final {
    auto var = GetRef<Var>(var_node);
    auto it = param_states_.find(var);
    if (it == param_states_.end()) {
      LOG(FATAL) << "Free variable " << var->name_hint();
      return {};
    } else {
      ICHECK(data_dependents_per_input_.size());
      auto data_dependent = data_dependents_per_input_.back();
      if (data_dependent) {
        param_states_[var] |= kNeedInputData;
        return param_data_[var];
      } else {
        param_states_[var] |= kNeedInputShape;
        return param_shapes_[var];
      }
    }
  }

  Array<te::Tensor> VisitExpr_(const ConstantNode* op) final {
    using tir::make_const;
    ICHECK(data_dependents_per_input_.size());
    bool data_dependent = data_dependents_per_input_.back();
    if (!op->is_scalar()) {
      // This is a constant weight, extract the shape of the weight tensor.
      // This can not be data dependent.
      CHECK(!data_dependent);
      auto ttype = op->checked_type().as<TensorTypeNode>();
      int ndim = static_cast<int>(ttype->shape.size());
      Array<PrimExpr> out_shape{ndim};
      te::Tensor value = tvm::te::compute(
          out_shape,
          [&](const Array<tvm::tir::Var>& indices) {
            auto idx = indices[0];
            PrimExpr ret = make_const(DataType::Int(64), 0);
            for (int i = 0; i < ndim; i++) {
              ret = tvm::if_then_else(idx == i, ttype->shape[i], ret);
            }
            return ret;
          },
          "shape_const", topi::kBroadcast);
      scalars_.push_back(value);
      return {value};
    }
    if (data_dependent) {
      void* data = op->data->data;
      DataType dtype = DataType(op->data->dtype);
      auto value = tvm::te::compute(
          {},
          [&](const Array<tvm::tir::Var>&) {
            if (dtype == DataType::Int(32)) {
              return make_const(dtype, static_cast<const int32_t*>(data)[0]);
            } else if (dtype == DataType::Int(64)) {
              return make_const(dtype, static_cast<const int64_t*>(data)[0]);
            } else if (dtype == DataType::Float(32)) {
              return make_const(dtype, static_cast<const float*>(data)[0]);
            } else if (dtype == DataType::Float(64)) {
              return make_const(dtype, static_cast<const double*>(data)[0]);
            } else if (dtype == DataType::Bool()) {
              return make_const(dtype, static_cast<const uint8_t*>(data)[0]);
            } else {
              LOG(FATAL) << "not handled";
              return tvm::PrimExpr();
            }
          },
          "data_const", topi::kBroadcast);
      scalars_.push_back(value);
      return {value};
    } else {
      auto value = tvm::te::compute(
          {}, [&](const Array<tvm::tir::Var>&) { return tir::make_const(DataType::Int(64), 0); },
          "shape_const", topi::kBroadcast);
      scalars_.push_back(value);
      return {value};
    }
  }

  Array<te::Tensor> VisitExpr_(const CallNode* call_node) final {
    static auto fshape_func = Op::GetAttrMap<FShapeFunc>("FShapeFunc");
    static auto tshape_data_dependent = Op::GetAttrMap<TShapeDataDependent>("TShapeDataDependent");
    ICHECK(call_node->op.as<OpNode>()) << "Primitive function only allows call into primitive ops";
    Op op = Downcast<Op>(call_node->op);
    ICHECK(data_dependents_per_input_.empty() || !data_dependents_per_input_.back())
        << "Error in op fusion: output of the shape func is fed to a "
        << "data-dependent shape func";
    ICHECK_GT(fshape_func.count(op), 0) << "Internal error, cannot find ShapeFunc for " << op->name;
    ICHECK_GT(tshape_data_dependent.count(op), 0)
        << "Internal error, cannot find TShapeDataDependent for " << op->name;

    Array<Integer> dep_spec = tshape_data_dependent[op];
    if (dep_spec.size() == 1) {
      // This is for cases when data dependence is specified per op
      // Replicate 0 or 1 flag to all arguments
      for (size_t i = 1; i < call_node->args.size(); ++i) {
        dep_spec.push_back(dep_spec[0]);
      }
    }

    // Visit all inputs
    Array<te::Tensor> inputs;
    int count_tuple = 0;
    for (size_t i = 0; i < call_node->args.size(); ++i) {
      Expr arg = call_node->args[i];
      if (arg->checked_type().as<TupleTypeNode>()) {
        ++count_tuple;
      }
      data_dependents_per_input_.push_back(dep_spec[i]->value != 0);
      for (te::Tensor tensor : VisitExpr(arg)) {
        inputs.push_back(tensor);
      }
      data_dependents_per_input_.pop_back();
    }
    if (count_tuple) {
      ICHECK_EQ(call_node->args.size(), 1U) << "Only allow function with a single tuple input";
    }
    // Get output ndims
    auto ret_type = call_node->checked_type();
    Array<IndexExpr> out_ndims;
    if (const auto* ttype = ret_type.as<TensorTypeNode>()) {
      out_ndims.push_back(IntImm(DataType::Int(32), ttype->shape.size()));
    } else {
      auto rtype = ret_type.as<TupleTypeNode>();
      // TODO(@icemelon): Allow recursive tuple
      ICHECK(rtype);
      for (size_t i = 0; i < rtype->fields.size(); ++i) {
        auto ttype = rtype->fields[i].as<TensorTypeNode>();
        ICHECK(ttype);
        out_ndims.push_back(IntImm(DataType::Int(32), ttype->shape.size()));
      }
    }
    // Call shape function
    auto outputs = fshape_func[op](call_node->attrs, inputs, out_ndims);
    readable_name_stream_ << "_" << op->name;
    return outputs;
  }

  Array<te::Tensor> VisitExpr_(const FunctionNode* op) final {
    LOG(FATAL) << "Do not support sub function";
    return Array<te::Tensor>();
  }

  Array<te::Tensor> VisitExpr_(const LetNode* op) final {
    Array<te::Tensor> val = VisitExpr(op->value);
    ICHECK(!memo_.count(op->var));
    memo_[op->var] = val;
    return VisitExpr(op->body);
  }

  Array<te::Tensor> VisitExpr_(const TupleNode* op) final {
    Array<te::Tensor> fields;
    for (Expr field : op->fields) {
      ICHECK(field->checked_type().as<TensorTypeNode>()) << "Only allow Tuple of Tensor";
      Array<te::Tensor> res = VisitExpr(field);
      ICHECK_EQ(res.size(), 1);
      fields.push_back(res[0]);
    }
    return fields;
  }

  Array<te::Tensor> VisitExpr_(const TupleGetItemNode* op) final {
    Array<te::Tensor> input_shapes = VisitExpr(op->tuple);
    Array<te::Tensor> out;
    out.push_back(input_shapes[op->index]);
    return out;
  }

 private:
  /*! \brief String stream for function name */
  std::ostringstream readable_name_stream_;
  /*! \brief Map from parameter to its shape function usage state */
  std::unordered_map<Expr, int, ObjectPtrHash, ObjectPtrEqual> param_states_;
  /*! \brief Map from parameter to list of data placeholder */
  std::unordered_map<Expr, Array<te::Tensor>, ObjectPtrHash, ObjectPtrEqual> param_data_;
  /*! \brief Map from parameter to list of shape placeholder */
  std::unordered_map<Expr, Array<te::Tensor>, ObjectPtrHash, ObjectPtrEqual> param_shapes_;
  /*! \brief Stack of data dependencies for shape function, specified per each op input */
  std::vector<bool> data_dependents_per_input_;
  /*! \brief Scalars used in the shape function */
  Array<te::Tensor> scalars_;
};

class TECompilerImpl : public TECompilerNode {
 public:
  // Lower the function.
  CachedFunc Lower(const CCacheKey& key) { return LowerInternal(key)->cached_func; }

  // For now, build one module per function.
  PackedFunc JIT(const CCacheKey& key) final {
    CCacheValue value = LowerInternal(key);
    if (value->packed_func != nullptr) return value->packed_func;
    // build the function.
    tvm::runtime::Module m;
    if (const auto* f = runtime::Registry::Get("relay.backend.build")) {
      m = (*f)(value->cached_func->funcs, key->target);
    } else {
      m = build(value->cached_func->funcs, key->target, Target(nullptr));
    }
    value->packed_func = m.GetFunction(value->cached_func->func_name);
    return value->packed_func;
  }

  CachedFunc LowerShapeFunc(const CCacheKey& key) final {
    return LowerShapeFuncInternal(key)->cached_func;
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
        ICHECK(code_gen.defined()) << "No external codegen is set";
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
        ICHECK(pf) << "Failed to find the codegen tool for " << ext_name << "\n";
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
      // if (!backend::IsTECompilerCacheDisabled()) {
        cache_[key] = value;
      // }
    }
    cur_ccache_key_ = key;

    // No need to lower external functions for now. We will invoke the external
    // codegen tool once and lower all functions together.
    if (key->source_func->GetAttr<String>(attr::kCompiler).defined()) {
      auto cache_node = make_object<CachedFuncNode>();
      const auto name_node = key->source_func->GetAttr<String>(tvm::attr::kGlobalSymbol);
      ICHECK(name_node.defined()) << "External function has not been attached a name yet.";
      cache_node->func_name = std::string(name_node.value());
      cache_node->target = Target("ext_dev");
      cache_node->funcs->Add(GlobalVar(cache_node->func_name), key->source_func);
      value->cached_func = CachedFunc(cache_node);
      return value;
    }
    // Enforce use the target.
    With<Target> target_scope(key->target);

    ICHECK(!value->cached_func.defined());
    auto cfunc = CreateSchedule(key->source_func, key->target);
    auto cache_node = make_object<CachedFuncNode>(*(cfunc.operator->()));

    // Skip lowering for device copy node.
    const Expr body = (key->source_func)->body;
    if (const CallNode* call_node = body.as<CallNode>()) {
      if (call_node->attrs.as<DeviceCopyAttrs>()) {
        value->cached_func = CachedFunc(cache_node);
        return value;
      }
    }

    cache_node->func_name = GetUniqueName(cache_node->func_name);
    // NOTE: array will copy on write.
    Array<te::Tensor> all_args = cache_node->inputs;
    for (te::Tensor arg : cache_node->outputs) {
      all_args.push_back(arg);
    }
    // lower the function
    if (const auto* f = runtime::Registry::Get("relay.backend.lower")) {
      cache_node->funcs = (*f)(cfunc->schedule, all_args, cache_node->func_name, key->source_func);
    } else {
      using tvm::transform::PassContext;
      With<PassContext> fresh_pass_ctx_scope(PassContext::Create());

      std::unordered_map<te::Tensor, tir::Buffer> binds;
      cache_node->funcs = tvm::lower(cfunc->schedule, all_args, cache_node->func_name, binds);
    }
    value->cached_func = CachedFunc(cache_node);
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
    auto spair = MakeShapeFunc().Create(key->source_func);
    auto cache_node = make_object<CachedFuncNode>(*(spair.second.operator->()));
    cache_node->func_name = GetUniqueName(cache_node->func_name);
    cache_node->target = key->target;

    Array<te::Tensor> all_args = cache_node->inputs;
    for (te::Tensor arg : cache_node->outputs) {
      all_args.push_back(arg);
    }

    using tvm::transform::PassContext;
    With<PassContext> fresh_pass_ctx_scope(PassContext::Create());

    std::unordered_map<te::Tensor, tir::Buffer> binds;
    cache_node->funcs = tvm::lower(spair.first, all_args, cache_node->func_name, binds);
    value->cached_func = CachedFunc(cache_node);
    return value;
  }
  /*!
   * \brief Get unique name from name.
   * \param name The orginal name.
   * \return Updated name which is unique.
   */
  std::string GetUniqueName(std::string name) {
    for (size_t i = 0; i < name.length(); ++i) {
      if (name[i] == '.') name[i] = '_';
    }
    while (true) {
      auto it = name_map_.find(name);
      if (it == name_map_.end()) {
        name_map_[name] = 1;
        return name;
      } else {
        std::ostringstream os;
        os << name << "_" << it->second;
        ++(it->second);
        name = os.str();
      }
    }
    return name;
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


TVM_REGISTER_NODE_TYPE(PrimFnCallAttrs);

class LowerTensorExpr : public ExprMutator {
 public:
  LowerTensorExpr(const IRModule& module, const TargetsMap& targets, TECompiler compiler)
    : module_(module), targets_(targets), compiler_(compiler) {}

  // /*!
  //  * \brief Add node to graph
  //  *
  //  * \param node
  //  * \param expr
  //  * \return std::vector<_NodeRef>
  //  */
  // std::vector<GraphNodeRef> AddNode(GraphObjectPtr node, Expr expr) {
  //   auto checked_type = expr->checked_type();
  //   size_t count = storage_device_map_.count(expr);
  //   ICHECK_GT(count, 0) << "Expr is not existing in storage plan";
  //   auto storage_device_info = storage_device_map_[expr];
  //   ICHECK_EQ(storage_device_info.size(), 2);
  //   // storage
  //   std::vector<int64_t> storage_info;
  //   for (auto& v : storage_device_info[0]) {
  //     storage_info.push_back(v->value);
  //   }
  //   node->attrs_["storage_id"] = std::move(storage_info);
  //   // type
  //   std::vector<int64_t> device_types;
  //   for (auto& v : storage_device_info[1]) {
  //     device_types.push_back(v->value);
  //   }
  //   size_t num_unknown_devices = std::count(device_types.begin(), device_types.end(), 0);
  //   if (num_unknown_devices != 0 && num_unknown_devices != device_types.size()) {
  //     LOG(FATAL) << "The graph contains not annotated nodes for "
  //                << "heterogeneous execution. All nodes must be "
  //                << "annotated.";
  //   }
  //   if (num_unknown_devices == 0) {
  //     node->attrs_["device_index"] = device_types;
  //   }
  //   auto node_id = nodes_.size();
  //   nodes_.push_back(node);
  //   // Tuple return value, flatten as tuple
  //   if (const auto* tuple_type = checked_type.as<TupleTypeNode>()) {
  //     std::vector<GraphNodeRef> ret;
  //     ShapeVector shape;
  //     std::vector<std::string> dtype;
  //     for (size_t i = 0; i < tuple_type->fields.size(); ++i) {
  //       if (const auto* typ = tuple_type->fields[i].as<TensorTypeNode>()) {
  //         ret.push_back(GraphNodeRef(node_id, i));
  //         shape.emplace_back(_ShapeToJSON(typ->shape));
  //         dtype.emplace_back(DType2String(typ->dtype));
  //       } else {
  //         LOG(FATAL) << "type " << checked_type->GetTypeKey() << " not supported";
  //       }
  //     }
  //     ICHECK_EQ(node->Type(), kGraphOpNode);
  //     auto op_nd = std::dynamic_pointer_cast<GraphOpNode>(node);
  //     op_nd->attrs_["shape"] = shape;
  //     op_nd->attrs_["dtype"] = dtype;
  //     op_nd->num_outputs_ = tuple_type->fields.size();
  //     return ret;
  //   }
  //   // Normal tensor return type
  //   if (const auto* tensor_type = checked_type.as<TensorTypeNode>()) {
  //     ShapeVector shape;
  //     std::vector<std::string> dtype;
  //     shape.emplace_back(_ShapeToJSON(tensor_type->shape));
  //     dtype.emplace_back(DType2String(tensor_type->dtype));
  //     node->attrs_["shape"] = shape;
  //     node->attrs_["dtype"] = dtype;
  //   } else {
  //     LOG(FATAL) << "type " << checked_type->GetTypeKey() << " not supported";
  //   }
  //   return {GraphNodeRef(node_id, 0)};
  // }

  // std::vector<GraphNodeRef> GraphAddCallNode(const CallNode* op, const std::string& op_name,
  //                                            const std::string& func_name) {
  //   std::vector<GraphNodeRef> inputs;
  //   for (auto arg : op->args) {
  //     auto res = VisitExpr(arg);
  //     for (auto nr : res) {
  //       inputs.push_back(nr);
  //     }
  //   }
  //   auto node = GraphOpNode::make_node_ptr(op_name, GraphAttrs(), func_name, inputs, GraphAttrs());
  //   return AddNode(node, GetRef<Expr>(op));
  // }

  Expr VisitExpr_(const CallNode* op) override {
    // Expr expr = GetRef<Expr>(op);
    // Function func;

    // if (op->op.as<OpNode>()) {
    //   LOG(FATAL) << "Operators should be transformed away; try applying"
    //              << "the fuse_ops transformation to the expression.";
    // } else if (op->op.as<GlobalVarNode>()) {
    //   LOG(FATAL) << "Not implemented";
    // } else if (op->op.as<FunctionNode>()) {
    //   func = GetRef<Function>(op->op.as<FunctionNode>());
    // } else {
    //   LOG(FATAL) << "TVM runtime does not support calls to " << op->op->GetTypeKey();
    // }

    // if (!func->HasNonzeroAttr(attr::kPrimitive)) {
    //   LOG(FATAL) << "TVM only support calls to primitive functions "
    //              << "(i.e functions composed of fusable operator invocations)";
    // }

    // Target target;
    // // Handle external function
    // if (func->GetAttr<String>(attr::kCompiler).defined()) {
    //   target = Target("ext_dev");
    //   CCacheKey key = CCacheKey(func, target);
    //   CachedFunc ext_func = compile_engine_->Lower(key);

    //   ICHECK(ext_func.defined()) << "External function is not defined.";
    //   UpdateConstants(func, &params_);
    //   return GraphAddCallNode(op, ext_func->func_name, ext_func->func_name);
    // }

    // ICHECK_GE(storage_device_map_.count(expr), 0);
    // auto& device_type = storage_device_map_[expr][1];
    // auto call_dev_type = device_type[0]->value;
    // // Normal Relay Function
    // if (targets_.size() == 1) {
    //   // homogeneous execution.
    //   const auto& it = targets_.begin();
    //   target = (*it).second;
    // } else {
    //   // heterogeneous execution.
    //   std::string call_dev_name;
    //   if (call_dev_type == 0) {
    //     call_dev_name = "llvm";
    //   } else {
    //     call_dev_name = runtime::DeviceName(call_dev_type);
    //   }
    //   if (targets_.count(call_dev_type) == 0) {
    //     LOG(FATAL) << "No target is provided for device " << call_dev_name;
    //   }
    //   target = targets_[call_dev_type];
    // }
    // CCacheKey key = CCacheKey(func, target);
    // CachedFunc lowered_func = compile_engine_->Lower(key);
    // if (!lowered_funcs_.count(target->str())) {
    //   lowered_funcs_[target->str()] = IRModule(Map<GlobalVar, BaseFunc>({}));
    // }
    // lowered_funcs_[target->str()]->Update(lowered_func->funcs);
    // return GraphAddCallNode(op, _GetUniqueName(lowered_func->func_name), lowered_func->func_name);
    LOG(FATAL) << "here";
  }

  // /*!
  //  * \brief Get unique name for func
  //  *
  //  * \param name
  //  * \return std::string
  //  */
  // std::string _GetUniqueName(const std::string& name) {
  //   if (!name_map_.count(name)) {
  //     name_map_[name] = 1;
  //     return name;
  //   }
  //   auto index = name_map_[name];
  //   name_map_[name] += 1;
  //   return _GetUniqueName(name + std::to_string(index));
  // }

  IRModule module_;
  TargetsMap targets_;
  TECompiler compiler_;
};

LoweredModule LowerTE(const IRModule& module, TargetsMap targets) {
  TECompiler compiler;

  auto pass = CreateFunctionPass([=](Function func, IRModule module, PassContext ctx) {
    LowerTensorExpr lower_te(module, targets, compiler);
    return Downcast<Function>(lower_te.VisitExpr(func));
  }, 0, "LowerTensorExpr", {});

  auto updated_module = pass(module);

  LoweredModule lowered_module;
  lowered_module.main_module = updated_module;
  return lowered_module;
}

bool PrimFnCallRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                    const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 4u);
  auto func_type = types[0].as<FuncTypeNode>();
  ICHECK(func_type != nullptr) << "input must be operator with known type";
  auto input_type = types[1].as<TupleTypeNode>();
  auto output_type = types[2].as<TupleTypeNode>();
  ICHECK(input_type != nullptr)
      << "internal invariant violated: invoke_tvm_op inputs must be a tuple";
  ICHECK(output_type != nullptr)
      << "internal invariant violated: invoke_tvm_op outputs must be a tuple";
  Type ex_output;
  if (func_type->ret_type.as<TensorTypeNode>()) {
    ex_output = TupleType({func_type->ret_type});
  } else {
    ICHECK(func_type->ret_type.as<TupleTypeNode>()) << "should be tuple type";
    ex_output = func_type->ret_type;
  }
  auto ex_input = TupleType(func_type->arg_types);
  reporter->Assign(ex_input, GetRef<Type>(input_type));
  reporter->Assign(ex_output, GetRef<Type>(output_type));
  reporter->Assign(types[3], TupleType::Empty());
  return true;
}

Call PrimFnCall(Expr func, Expr inputs, GlobalVar prim_fn_name) {
    auto attrs = make_object<PrimFnCallAttrs>();
    attrs->prim_fn = prim_fn_name;
    return Call(Op::Get("prim_fn_call"), {func, inputs}, Attrs());
}

// TVM_REGISTER_PASS_CONFIG_OPTION("relay.backend.use_auto_scheduler", Bool);
// TVM_REGISTER_PASS_CONFIG_OPTION("relay.backend.disable_compile_engine_cache", Bool);

// TVM_REGISTER_GLOBAL("relay.backend._make_LoweredOutput")
//     .set_body_typed([](tvm::Array<te::Tensor> outputs, OpImplementation impl) {
//       return LoweredOutput(outputs, impl);
//     });

// TVM_REGISTER_GLOBAL("relay.backend._make_CCacheKey")
//     .set_body_typed([](Function source_func, Target target) {
//       return CCacheKey(source_func, target);
//     });

// TVM_REGISTER_GLOBAL("relay.backend._TECompilerGlobal").set_body_typed([]() {
//   return TECompiler::Global();
// });

// TVM_REGISTER_GLOBAL("relay.backend._TECompilerClear").set_body_typed([](TECompiler self) {
//   self->Clear();
// });

// TVM_REGISTER_GLOBAL("relay.backend._TECompilerLower")
//     .set_body_typed([](TECompiler self, CCacheKey key) { return self->Lower(key); });

// TVM_REGISTER_GLOBAL("relay.backend._TECompilerLowerShapeFunc")
//     .set_body_typed([](TECompiler self, CCacheKey key) { return self->LowerShapeFunc(key); });

// TVM_REGISTER_GLOBAL("relay.backend._CompileLowerExternalFunctions")
//     .set_body_typed([](TECompiler self) { return self->LowerExternalFunctions(); });

// TVM_REGISTER_GLOBAL("relay.backend._TECompilerJIT")
//     .set_body_typed([](TECompiler self, CCacheKey key) { return self->JIT(key); });

// TVM_REGISTER_GLOBAL("relay.backend._TECompilerListItems").set_body_typed([](TECompiler self) {
//   TECompilerImpl* ptr = dynamic_cast<TECompilerImpl*>(self.operator->());
//   ICHECK(ptr != nullptr);
//   return ptr->ListItems();
// });

// TVM_REGISTER_GLOBAL("relay.backend._TECompilerGetCurrentCCacheKey")
//     .set_body_typed([](TECompiler self) {
//       TECompilerImpl* ptr = dynamic_cast<TECompilerImpl*>(self.operator->());
//       ICHECK(ptr != nullptr);
//       return ptr->GetCurrentCCacheKey();
//     });

}  // namespace tirc
}  // namespace relay
}  // namespace tvm
