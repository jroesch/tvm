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
 * \file relay/backend/graph_codegen.cc
 * \brief Graph executor codegen
 */

#include <dmlc/any.h>
#include <dmlc/json.h>
#include <tvm/ir/module.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/object.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/function.h>

#include <list>
#include <string>
#include <vector>

#include "compile_engine.h"
#include "te_compiler.h"
#include "utils.h"

namespace tvm {
namespace relay {

/// TODO(@jroesch, @csullivan): declare directly elsewhere
Map<Expr, Array<Array<tvm::Integer>>> GraphPlanMemory(const Function& func);

namespace backend {

class GraphNode;
class GraphInputNode;
class GraphOpNode;

using IntegerArray = Array<Integer>;
using ShapeVector = std::vector<std::vector<int64_t>>;
using GraphAttrs = std::unordered_map<std::string, dmlc::any>;
using GraphObjectPtr = std::shared_ptr<GraphNode>;
using GraphInputObjectPtr = std::shared_ptr<GraphInputNode>;
using GraphOpObjectPtr = std::shared_ptr<GraphOpNode>;
using TargetsMap = std::unordered_map<int, Target>;

/*! \brief Node types */
enum GraphNodeType {
  kGraphNop,
  kGraphInputNode,
  kGraphOpNode,
};

class GraphNodeRef {
 public:
  GraphNodeRef() {}
  GraphNodeRef(int ident, int index, int version = 0)
      : ident_(ident), index_(index), version_(version) {}

  inline void Save(dmlc::JSONWriter* writer) const {
    writer->BeginArray();
    writer->WriteArrayItem(ident_);
    writer->WriteArrayItem(index_);
    writer->WriteArrayItem(version_);
    writer->EndArray();
  }

  inline void Load(dmlc::JSONReader* reader) { LOG(FATAL) << "Not implemented."; }

 protected:
  int ident_;
  int index_{0};
  int version_{0};
};

/*! \brief Base Node class */
class GraphNode {
 public:
  GraphNode() {}
  virtual void Save(dmlc::JSONWriter* writer) const {}
  virtual void Load(dmlc::JSONReader* reader) {}
  virtual GraphNodeType Type() const { return kGraphNop; }
  virtual ~GraphNode() {}

 public:
  int num_outputs_{1};
  std::string name_;
  GraphAttrs attrs_;
};

/*! \brief Input Node */
class GraphInputNode : public GraphNode {
 public:
  GraphInputNode() {}
  GraphInputNode(const std::string& name, const GraphAttrs& attrs) {
    name_ = name;
    attrs_ = attrs;
  }

  GraphNodeType Type() const override { return kGraphInputNode; }

  void Save(dmlc::JSONWriter* writer) const override {
    const std::string op_name{"null"};
    writer->BeginObject();
    writer->WriteObjectKeyValue("op", op_name);
    writer->WriteObjectKeyValue("name", this->name_);
    writer->WriteObjectKeyValue("inputs", std::list<int>());
    writer->EndObject();
  }
  static std::shared_ptr<GraphNode> make_node_ptr(const std::string& name,
                                                  const GraphAttrs& attrs) {
    auto ptr = std::make_shared<GraphInputNode>(name, attrs);
    return std::dynamic_pointer_cast<GraphNode>(ptr);
  }
};

/*! \brief Op Node */
class GraphOpNode : public GraphNode {
 public:
  GraphOpNode() {}
  GraphOpNode(const std::string& name, const GraphAttrs& nd_attrs, const std::string& op_name,
              const std::vector<GraphNodeRef>& inputs, const GraphAttrs& attrs,
              size_t num_outputs = 1) {
    name_ = name;
    attrs_ = nd_attrs;
    op_name_ = op_name;
    inputs_ = inputs;
    op_attrs_ = attrs;
    num_outputs_ = num_outputs;
    op_attrs_["func_name"] = op_name_;
    op_attrs_["flatten_data"] = std::string("0");
    op_attrs_["num_inputs"] = std::to_string(inputs_.size());
    op_attrs_["num_outputs"] = std::to_string(num_outputs_);
  }

  GraphNodeType Type() const override { return kGraphOpNode; }

  void Save(dmlc::JSONWriter* writer) const override {
    GraphAttrs attrs = op_attrs_;
    attrs["func_name"] = this->op_name_;
    attrs["flatten_data"] = std::string("0");
    attrs["num_inputs"] = std::to_string(this->inputs_.size());
    attrs["num_outputs"] = std::to_string(this->num_outputs_);
    writer->BeginObject();
    writer->WriteObjectKeyValue("op", op_type_name_);
    writer->WriteObjectKeyValue("name", name_);
    writer->WriteObjectKeyValue("attrs", attrs);
    writer->WriteObjectKeyValue("inputs", this->inputs_);
    writer->EndObject();
  }
  static std::shared_ptr<GraphNode> make_node_ptr(const std::string& name,
                                                  const GraphAttrs& nd_attrs,
                                                  const std::string& op_name,
                                                  const std::vector<GraphNodeRef>& inputs,
                                                  const GraphAttrs& attrs, size_t num_outputs = 1) {
    auto ptr = std::make_shared<GraphOpNode>(name, nd_attrs, op_name, inputs, attrs, num_outputs);
    return std::dynamic_pointer_cast<GraphNode>(ptr);
  }

 public:
  std::string op_name_;
  std::vector<GraphNodeRef> inputs_;
  GraphAttrs op_attrs_;

 private:
  const std::string op_type_name_{"tvm_op"};
};

/*! \brief Code generator for the graph executor, produces a module containing the graph JSON,
 * module, and parameters.
 */
class GraphExecutorCodegen : public backend::MemoizedExprTranslator<std::vector<GraphNodeRef>> {
 public:
  GraphExecutorCodegen(runtime::Module* mod, const TargetsMap& targets) : mod_(mod) {
    targets_ = targets;
  }

  /*!
   * \brief Update the "main" control function's metadata
   *
   * \param func The main function that contains calls to relay primitive functions
   */
  void UpdateMainWorkspaceSize(const Function& func) {
    // This is a Map<device,Map<storage_id, size>>
    std::unordered_map<int, std::unordered_map<int, int>> sid_workspace;
    // This is a Map<device, size_of_inputs_and_outputs>
    std::unordered_map<int, int> device_io;
    // This is a Map<device, size_of_constants>
    std::unordered_map<int, int> device_consts;

    // Initialize the maps to zero
    for (const auto& kv : storage_device_map_) {
      auto sids = kv.second[0];
      auto devices = kv.second[1];
      CHECK_EQ(sids.size(), devices.size());
      for (uint32_t i = 0; i < sids.size(); i++) {
        sid_workspace[devices[i]][sids[i]] = 0;
        device_io[devices[i]] = 0;
        device_consts[devices[i]] = 0;
      }
    }

    // Collect sizes of tensors
    for (const auto& kv : storage_device_map_) {
      auto size_bytes = CalculateRelayExprSizeBytes(kv.first->checked_type());
      auto sids = kv.second[0];
      auto devices = kv.second[1];
      if (kv.first->IsInstance<ConstantNode>()) {
        for (const auto& dev : devices) {
          device_consts[dev] += size_bytes;
        }
        continue;
      } else if (kv.first->IsInstance<VarNode>() || kv.first == func->body) {
        for (const auto& dev : devices) {
          device_io[dev] += size_bytes;
        }
        continue;
      }
      for (uint32_t i = 0; i < sids.size(); i++) {
        // Here we record the largest size of the tensor
        // that share the same storage id, because storage_id will
        // be shared between multiple tensors that are not live simultaneously.
        if (size_bytes > sid_workspace[devices[i]][sids[i]]) {
          sid_workspace[devices[i]][sids[i]] = size_bytes;
        }
      }
    }

    // This is a Map<device, workspace_size>
    std::unordered_map<int, int> device_workspace;
    // Once we know the sizes of sids, we need to accumulate per device
    for (const auto& dev_sid_size : sid_workspace) {
      auto dev = dev_sid_size.first;
      device_workspace[dev] = 0;
      for (const auto& sid_size : dev_sid_size.second) {
        device_workspace[dev] += sid_size.second;
      }
    }

    // Populate FunctionInfo
    auto fi_node = make_object<FunctionInfoNode>();
    // Initialize all target workspaces to zero
    for (const auto& kv : targets_) {
      auto tgt = kv.second;
      fi_node->workspace_sizes.Set(tgt, 0);
    }
    for (const auto& dev_and_size : device_workspace) {
      auto tgt = GetTargetFromInteger(dev_and_size.first);
      fi_node->workspace_sizes.Set(tgt, dev_and_size.second);
      fi_node->relay_primfuncs.Set(tgt, func);
    }
    for (const auto& dev_and_size : device_io) {
      auto tgt = GetTargetFromInteger(dev_and_size.first);
      fi_node->io_sizes.Set(tgt, dev_and_size.second);
    }
    for (const auto& dev_and_size : device_consts) {
      auto tgt = GetTargetFromInteger(dev_and_size.first);
      fi_node->constant_sizes.Set(tgt, dev_and_size.second);
    }

    function_metadata_.Set(String(runtime::symbol::tvm_module_main), FunctionInfo(fi_node));
  }

  LoweredOutput Codegen(relay::Function func) {
    storage_device_map_ = GraphPlanMemory(func);

    // Andrew why is this in here?
    //
    // UpdateMainWorkspaceSize(func);

    // This first phase moves from implicit use of compile engine,
    // to instead explicitly lowering the incoming IRModule, and then
    // performing the preexisting graph executor code generation phase.
    IRModule mod = IRModule::FromExpr(func);

    // Build a map from each operation to device.
    tec::DeviceMap device_context_map;
    for (const auto& it : storage_device_map_) {
      auto expr = it.first;
      auto storage_and_device = it.second;
      ICHECK_EQ(storage_and_device.size(), 2u);
      auto device_type = storage_and_device[1];
      tvm::Device dev;
      dev.device_id = 0;
      dev.device_type = static_cast<DLDeviceType>(device_type[0]->value);
      device_context_map.insert({expr, dev});
    }

    // todo map targets down
    auto lowered_module = tec::LowerTE(mod, targets_, device_context_map, [this](Function func) {
      // We need to maintain the constant map for external functions so we pass this
      // processing function which allows us to process each function as we lower it.
      if (func->GetAttr<String>(attr::kCompiler).defined()) {
        UpdateConstants(func, &params_);
      }
    });

    auto main_module = lowered_module.main_module;
    main_module = relay::transform::InferType()(main_module);
    relay::Function main_func = Downcast<relay::Function>(main_module->Lookup("main"));

    // Now that we have lowered all operators to TIR code, we can proceed with compilation.
    storage_device_map_ = GraphPlanMemory(main_func);

    // First we convert all the parameters into input nodes.
    for (auto param : main_func->params) {
      auto node_ptr = GraphInputNode::make_node_ptr(param->name_hint(), GraphAttrs());
      var_map_[param.get()] = AddNode(node_ptr, param);
    }

    heads_ = VisitExpr(main_func->body);
    std::ostringstream os;

    dmlc::JSONWriter writer(&os);
    GetJSON(&writer);
    LoweredOutput ret;
    ret.graph_json = os.str();
    ret.params = std::unordered_map<std::string, std::pair<int, const tvm::runtime::NDArray>>();
    for (auto param : params_) {
      ret.params.emplace(std::make_pair(
          param.first,
          std::make_pair(static_cast<int>(param_storage_ids_[param.first]), param.second)));
    }

    ret.function_metadata = std::move(function_metadata_);
    ret.lowered_funcs = lowered_module.per_target_module;
    ret.external_mods = lowered_module.external_mods;
    return ret;
  }

 protected:
  /*!
   * \brief Extract shape from expr to vector<int64_t>
   *
   * \param shape
   * \return std::vector<int64_t>
   */
  std::vector<int64_t> _ShapeToJSON(tvm::Array<IndexExpr> shape) {
    std::vector<int64_t> ret;
    for (IndexExpr dim : shape) {
      const int64_t* pval = tir::as_const_int(dim);
      ret.push_back(*pval);
    }
    return ret;
  }

  /*!
   * \brief Add node to graph
   *
   * \param node
   * \param expr
   * \return std::vector<_NodeRef>
   */
  std::vector<GraphNodeRef> AddNode(GraphObjectPtr node, Expr expr) {
    auto checked_type = expr->checked_type();
    size_t count = storage_device_map_.count(expr);
    ICHECK_GT(count, 0) << "Expr is not existing in storage plan";
    auto storage_device_info = storage_device_map_[expr];
    ICHECK_EQ(storage_device_info.size(), 3);
    // storage
    std::vector<int64_t> storage_info;
    for (auto& v : storage_device_info[0]) {
      storage_info.push_back(v->value);
    }
    node->attrs_["storage_id"] = std::move(storage_info);
    // type
    std::vector<int64_t> device_types;
    for (auto& v : storage_device_info[1]) {
      device_types.push_back(v->value);
    }
    size_t num_unknown_devices = std::count(device_types.begin(), device_types.end(), 0);
    if (num_unknown_devices != 0 && num_unknown_devices != device_types.size()) {
      LOG(FATAL) << "The graph contains not annotated nodes for "
                 << "heterogeneous execution. All nodes must be "
                 << "annotated.";
    }
    if (num_unknown_devices == 0) {
      node->attrs_["device_index"] = device_types;
    }
    auto node_id = nodes_.size();
    nodes_.push_back(node);
    // Tuple return value, flatten as tuple
    if (const auto* tuple_type = checked_type.as<TupleTypeNode>()) {
      std::vector<GraphNodeRef> ret;
      ShapeVector shape;
      std::vector<std::string> dtype;
      for (size_t i = 0; i < tuple_type->fields.size(); ++i) {
        if (const auto* typ = tuple_type->fields[i].as<TensorTypeNode>()) {
          ret.push_back(GraphNodeRef(node_id, i));
          shape.emplace_back(_ShapeToJSON(typ->shape));
          dtype.emplace_back(DType2String(typ->dtype));
        } else {
          LOG(FATAL) << "type " << checked_type->GetTypeKey() << " not supported";
        }
      }
      ICHECK_EQ(node->Type(), kGraphOpNode);
      auto op_nd = std::dynamic_pointer_cast<GraphOpNode>(node);
      op_nd->attrs_["shape"] = shape;
      op_nd->attrs_["dtype"] = dtype;
      op_nd->num_outputs_ = tuple_type->fields.size();
      return ret;
    }
    // Normal tensor return type
    if (const auto* tensor_type = checked_type.as<TensorTypeNode>()) {
      ShapeVector shape;
      std::vector<std::string> dtype;
      shape.emplace_back(_ShapeToJSON(tensor_type->shape));
      dtype.emplace_back(DType2String(tensor_type->dtype));
      node->attrs_["shape"] = shape;
      node->attrs_["dtype"] = dtype;
    } else {
      LOG(FATAL) << "type " << checked_type->GetTypeKey() << " not supported";
    }
    return {GraphNodeRef(node_id, 0)};
  }

  std::vector<GraphNodeRef> VisitExpr_(const VarNode* op) override {
    Expr expr = GetRef<Expr>(op);
    return var_map_[expr.get()];
  }

  std::vector<GraphNodeRef> VisitExpr_(const ConstantNode* op) override {
    Expr expr = GetRef<Expr>(op);
    size_t index = params_.size();
    std::string name = "p" + std::to_string(index);
    auto node = GraphInputNode::make_node_ptr(name, GraphAttrs());
    auto to_return = AddNode(node, expr);
    CHECK_EQ(to_return.size(), 1) << "Expected exactly 1 parameter node created";
    param_storage_ids_[name] = storage_device_map_[expr][0][0]->value;
    params_[name] = op->data;
    return to_return;
  }

  std::vector<GraphNodeRef> VisitExpr_(const TupleNode* op) override {
    std::vector<GraphNodeRef> fields;
    for (auto field : op->fields) {
      auto ref_vec = VisitExpr(field);
      for (auto ref : ref_vec) {
        fields.push_back(ref);
      }
    }
    return fields;
  }

  std::vector<GraphNodeRef> GraphAddCallNode(const CallNode* op, const std::string& op_name,
                                             const std::string& func_name, GraphAttrs attrs) {
    std::vector<GraphNodeRef> inputs;
    for (auto arg : op->args) {
      auto res = VisitExpr(arg);
      for (auto nr : res) {
        inputs.push_back(nr);
      }
    }
    auto node = GraphOpNode::make_node_ptr(op_name, GraphAttrs(), func_name, inputs, attrs);
    return AddNode(node, GetRef<Expr>(op));
  }

  std::vector<GraphNodeRef> VisitExpr_(const CallNode* call_node) override {
    relay::Call call = GetRef<Call>(call_node);
    if (auto global_node = call->op.as<GlobalVarNode>()) {
      auto prim_fn_name = global_node->name_hint;

      Target target;

      ICHECK_GE(storage_device_map_.count(call), 0)
          << "Could not find a storage device for " << prim_fn_name
          << "The memory planning was either not performed for this precise node, or there is bug "
             "in the memory planner.";

      auto& device_type = storage_device_map_[call][1];
      auto call_dev_type = device_type[0]->value;
      // Normal Relay Function
      if (targets_.size() == 1) {
        // homogeneous execution.
        const auto& it = targets_.begin();
        target = (*it).second;
      } else {
        // heterogeneous execution.
        std::string call_dev_name;
        if (call_dev_type == 0) {
          call_dev_name = "llvm";
        } else {
          call_dev_name = runtime::DeviceName(call_dev_type);
        }
        if (targets_.count(call_dev_type) == 0) {
          LOG(FATAL) << "No target is provided for device " << call_dev_name;
        }
        target = targets_[call_dev_type];
      }

      return GraphAddCallNode(call_node, _GetUniqueName(prim_fn_name), prim_fn_name);
    } else {
      ICHECK(false) << "Non-primitive-call nodes should have been transformed away.\n"
                    << "The graph executor code generator expects all calls to have their callee "
                       "normalized to a GlobalVar but found a "
                    << call->GetTypeKey() << "."
                    << "AST: " << PrettyPrint(call) << PrettyPrint(call) << std::endl;
      return {};
    }
  }

  std::vector<GraphNodeRef> VisitExpr_(const LetNode* op) override {
    ICHECK_EQ(var_map_.count(op->var.get()), 0);
    var_map_[op->var.get()] = VisitExpr(op->value);
    return VisitExpr(op->body);
  }
  std::vector<GraphNodeRef> VisitExpr_(const TupleGetItemNode* op) override {
    auto vtuple = VisitExpr(op->tuple);
    return {vtuple[op->index]};
  }
  std::vector<GraphNodeRef> VisitExpr_(const OpNode* op) override {
    throw std::runtime_error("can not compile op in non-eta expanded form");
    return {};
  }
  std::vector<GraphNodeRef> VisitExpr_(const GlobalVarNode* op) override {
    throw std::runtime_error("");
    return {};
  }
  std::vector<GraphNodeRef> VisitExpr_(const IfNode* op) override {
    throw std::invalid_argument("if not supported");
    return {};
  }
  std::vector<GraphNodeRef> VisitExpr_(const FunctionNode* op) override {
    ICHECK(op->GetAttr<String>(attr::kCompiler).defined())
        << "Only functions supported by custom codegen";
    return {};
  }
  std::vector<GraphNodeRef> VisitExpr_(const RefCreateNode* op) override {
    throw std::invalid_argument("reference not supported");
    return {};
  }
  std::vector<GraphNodeRef> VisitExpr_(const RefReadNode* op) override {
    throw std::invalid_argument("reference not supported");
    return {};
  }
  std::vector<GraphNodeRef> VisitExpr_(const RefWriteNode* op) override {
    throw std::invalid_argument("reference not supported");
    return {};
  }
  std::vector<GraphNodeRef> VisitExpr_(const ConstructorNode* op) override {
    throw std::invalid_argument("ADT constructor case not yet implemented");
    return {};
  }
  std::vector<GraphNodeRef> VisitExpr_(const MatchNode* op) override {
    throw std::invalid_argument("match case not yet implemented");
    return {};
  }
  /*!
   * \brief Generate Graph JSON
   *
   * \param writer json writer
   */
  void GetJSON(dmlc::JSONWriter* writer) {
    std::vector<size_t> arg_nodes;
    for (size_t i = 0; i < nodes_.size(); ++i) {
      auto node = nodes_[i];
      if (node->Type() == kGraphInputNode) {
        arg_nodes.push_back(i);
      }
    }
    size_t num_entry = 0;
    ShapeVector shapes;
    std::vector<size_t> storage_ids;
    std::vector<size_t> device_types;
    std::vector<std::string> dltypes;
    std::vector<size_t> node_row_ptr{0};
    for (auto node : nodes_) {
      const auto& shape_vec = dmlc::get<ShapeVector>(node->attrs_["shape"]);
      const auto& storage_id = dmlc::get<std::vector<int64_t>>(node->attrs_["storage_id"]);
      const auto& dtype_vec = dmlc::get<std::vector<std::string>>(node->attrs_["dtype"]);

      ICHECK_EQ(node->num_outputs_, shape_vec.size());
      num_entry += node->num_outputs_;

      shapes.insert(shapes.end(), shape_vec.begin(), shape_vec.end());
      dltypes.insert(dltypes.end(), dtype_vec.begin(), dtype_vec.end());
      storage_ids.insert(storage_ids.end(), storage_id.begin(), storage_id.end());
      if (node->attrs_.count("device_index")) {
        const auto& dev_types = dmlc::get<std::vector<int64_t>>(node->attrs_["device_index"]);
        device_types.insert(device_types.end(), dev_types.begin(), dev_types.end());
      }
      node_row_ptr.push_back(num_entry);
    }
    writer->BeginObject();
    writer->WriteObjectKeyValue("nodes", nodes_);
    writer->WriteObjectKeyValue("arg_nodes", arg_nodes);
    writer->WriteObjectKeyValue("heads", heads_);
    std::unordered_map<std::string, std::vector<dmlc::any>> attrs;
    attrs["shape"].emplace_back(std::string("list_shape"));
    attrs["shape"].emplace_back(shapes);
    attrs["storage_id"].emplace_back(std::string("list_int"));
    attrs["storage_id"].emplace_back(storage_ids);
    if (device_types.size()) {
      attrs["device_index"].emplace_back(std::string("list_int"));
      attrs["device_index"].emplace_back(device_types);
    }
    attrs["dltype"].emplace_back(std::string("list_str"));
    attrs["dltype"].emplace_back(dltypes);
    writer->WriteObjectKeyValue("attrs", attrs);
    writer->WriteObjectKeyValue("node_row_ptr", node_row_ptr);
    writer->EndObject();
  }

  /*!
   * \brief Get unique name for func
   *
   * \param name
   * \return std::string
   */
  std::string _GetUniqueName(const std::string& name) {
    if (!name_map_.count(name)) {
      name_map_[name] = 1;
      return name;
    }
    auto index = name_map_[name];
    name_map_[name] += 1;
    return _GetUniqueName(name + std::to_string(index));
  }

 protected:
  /*! \brief nodes */
  std::vector<GraphObjectPtr> nodes_;
  /*! \brief output of graph */
  std::vector<GraphNodeRef> heads_;
  /*! \brief mod */
  runtime::Module* mod_;
  /*! \brief variable map */
  std::unordered_map<const Object*, std::vector<GraphNodeRef>> var_map_;
  /*! \brief target device */
  TargetsMap targets_;
  /*!
   * \brief parameters (i.e. ConstantNodes found in the graph).
   * These are take as inputs to the GraphExecutor.
   * Maps param name to a pair of storage_id and NDArray. At runtime, the storage_id can be
   * used to lookup the parameter.
   */
  std::unordered_map<std::string, runtime::NDArray> params_;
  std::unordered_map<std::string, int64_t> param_storage_ids_;
  /*! \brief plan memory of device result */
  Map<Expr, Array<IntegerArray>> storage_device_map_;
  /*! \brief lowered funcs */
  std::unordered_map<std::string, IRModule> lowered_funcs_;
  /*! \brief lowered funcs */
  Map<String, FunctionInfo> function_metadata_;
  /*! \brief name map */
  std::unordered_map<std::string, size_t> name_map_;
};

class GraphExecutorCodegenModule : public runtime::ModuleNode {
 public:
  GraphExecutorCodegenModule() {}
  virtual PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) {
    if (name == "init") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        ICHECK_EQ(args.num_args, 2) << "The expected of arguments are: "
                                    << "runtime::Module mod and Map<int, Target> targets";
        void* mod = args[0];
        Map<Integer, tvm::Target> tmp = args[1];
        TargetsMap targets;
        for (const auto& it : tmp) {
          auto dev_type = it.first.as<tir::IntImmNode>();
          ICHECK(dev_type);
          targets[dev_type->value] = it.second;
        }
        codegen_ = std::make_shared<GraphExecutorCodegen>(reinterpret_cast<runtime::Module*>(mod),
                                                          targets);
      });
    } else if (name == "codegen") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        Function func = args[0];
        this->output_ = this->codegen_->Codegen(func);
      });
    } else if (name == "get_graph_json") {
      return PackedFunc(
          [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->output_.graph_json; });
    } else if (name == "list_params_name") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        Array<runtime::String> ret;
        for (const auto& kv : this->output_.params) {
          ret.push_back(kv.first);
        }
        *rv = ret;
      });
    } else if (name == "get_param_by_name") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        String key = args[0];
        auto it = this->output_.params.find(key);
        CHECK(it != this->output_.params.end()) << "no such parameter " << key;
        *rv = (*it).second.second;
      });
    } else if (name == "get_param_id") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        String key = args[0];
        auto it = this->output_.params.find(key);
        CHECK(it != this->output_.params.end()) << "no such parameter " << key;
        *rv = (*it).second.first;
      });
    } else if (name == "get_irmodule") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        *rv = this->output_.lowered_funcs;
      });
    } else if (name == "get_external_modules") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        *rv = this->output_.external_mods;
      });
    } else if (name == "get_metadata") {
      return PackedFunc(
          [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->output_.metadata; });
    } else if (name == "get_function_metadata") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        *rv = this->output_.function_metadata;
      });
    } else {
      return PackedFunc([](TVMArgs args, TVMRetValue* rv) {});
    }
  }

  const char* type_key() const final { return "RelayGraphExecutorCodegenModule"; }

 private:
  std::shared_ptr<GraphExecutorCodegen> codegen_;
  LoweredOutput output_;
};

runtime::Module CreateGraphCodegenMod() {
  auto ptr = make_object<GraphExecutorCodegenModule>();
  return runtime::Module(ptr);
}

TVM_REGISTER_GLOBAL("relay.build_module._GraphExecutorCodegen")
    .set_body([](TVMArgs args, TVMRetValue* rv) { *rv = CreateGraphCodegenMod(); });

}  // namespace backend
}  // namespace relay
}  // namespace tvm

namespace dmlc {
namespace json {
// JSON utils
template <typename T>
inline bool SameType(const dmlc::any& data) {
  return std::type_index(data.type()) == std::type_index(typeid(T));
}

template <>
struct Handler<std::shared_ptr<tvm::relay::backend::GraphNode>> {
  inline static void Write(dmlc::JSONWriter* writer,
                           const std::shared_ptr<tvm::relay::backend::GraphNode>& data) {
    data->Save(writer);
  }
  inline static void Read(dmlc::JSONReader* reader,
                          std::shared_ptr<tvm::relay::backend::GraphNode>* data) {
    LOG(FATAL) << "Not implemented.";
  }
};
template <>
struct Handler<std::unordered_map<std::string, dmlc::any>> {
  inline static void Write(dmlc::JSONWriter* writer,
                           const std::unordered_map<std::string, dmlc::any>& data) {
    writer->BeginObject();
    for (const auto& kv : data) {
      auto k = kv.first;
      const dmlc::any& v = kv.second;
      if (SameType<std::string>(v)) {
        writer->WriteObjectKeyValue(k, dmlc::get<std::string>(v));
      } else if (SameType<int>(v)) {
        writer->WriteObjectKeyValue(k, dmlc::get<int>(v));
      } else if (SameType<std::vector<size_t>>(v)) {
        writer->WriteObjectKeyValue(k, dmlc::get<std::vector<size_t>>(v));
      } else if (SameType<std::vector<std::vector<int64_t>>>(v)) {
        writer->WriteObjectKeyValue(k, dmlc::get<std::vector<std::vector<int64_t>>>(v));
      } else if (SameType<std::vector<std::string>>(v)) {
        writer->WriteObjectKeyValue(k, dmlc::get<std::vector<std::string>>(v));
      } else if (SameType<std::vector<dmlc::any>>(v)) {
        writer->WriteObjectKeyValue(k, dmlc::get<std::vector<dmlc::any>>(v));
      } else {
        LOG(FATAL) << "Not supported";
      }
    }
    writer->EndObject();
  }
  inline static void Read(dmlc::JSONReader* reader,
                          std::unordered_map<std::string, dmlc::any>* data) {
    LOG(FATAL) << "Not implemented.";
  }
};

template <>
struct Handler<std::vector<dmlc::any>> {
  inline static void Write(dmlc::JSONWriter* writer, const std::vector<dmlc::any>& data) {
    writer->BeginArray();
    for (const auto& v : data) {
      if (SameType<std::string>(v)) {
        writer->WriteArrayItem(dmlc::get<std::string>(v));
      } else if (SameType<int>(v)) {
        writer->WriteArrayItem(dmlc::get<int>(v));
      } else if (SameType<std::vector<size_t>>(v)) {
        writer->WriteArrayItem(dmlc::get<std::vector<size_t>>(v));
      } else if (SameType<std::vector<std::vector<int64_t>>>(v)) {
        writer->WriteArrayItem(dmlc::get<std::vector<std::vector<int64_t>>>(v));
      } else if (SameType<std::vector<std::string>>(v)) {
        writer->WriteArrayItem(dmlc::get<std::vector<std::string>>(v));
      } else {
        LOG(FATAL) << "Not supported";
      }
    }
    writer->EndArray();
  }
  inline static void Read(dmlc::JSONReader* reader, std::vector<dmlc::any>* data) {
    LOG(FATAL) << "Not implemented.";
  }
};
}  // namespace json
}  // namespace dmlc
