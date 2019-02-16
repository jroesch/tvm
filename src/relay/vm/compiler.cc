/*!
 *  Copyright (c) 2018 by Contributors
 * \file tvm/relay/runtime/runtime.h
 * \brief Abstract device memory management API
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

using TagMap = std::unordered_map<tvm::relay::Constructor, size_t, NodeHash, NodeEqual>;
using TagNameMap = std::unordered_map<size_t, tvm::relay::Constructor>;
using GlobalMap = std::unordered_map<GlobalVar, size_t, NodeHash, NodeEqual>;
using ConstMap = std::unordered_map<Constant, size_t, NodeHash, NodeEqual>;

struct VMCompilerContext {
  Module module;
  TagNameMap tag_index_map;
  TagMap tag_map;
  GlobalMap global_map;
  ConstMap const_map;
  std::vector<LoweredFunc> lowered_funcs;
};

// Compute the constant pool, i.e a mapping from Constant node to constant index.
struct ConstantPool : ExprVisitor {
  std::set<GlobalVar> visited;
  Module module;
  ConstMap const_map;
  size_t index;

  ConstantPool(const Module& mod) :
    module(mod), const_map(), index(0) {}

  void VisitExpr_(const GlobalVarNode* var_node) {
    auto gvar = GetRef<GlobalVar>(var_node);
    if (visited.find(gvar) == visited.end()) {

      visited.insert(gvar);
      this->VisitExpr(this->module->Lookup(gvar));
    }
  }

  void VisitExpr_(const ConstantNode* const_node) {
    auto konst = GetRef<Constant>(const_node);
    auto it = this->const_map.find(konst);
    if (it == this->const_map.end()) {
      this->const_map.insert({ konst, index++ });
    }
  }
};

ConstMap LayoutConstantPool(const Module& module) {
  auto cp = ConstantPool(module);
  cp.VisitExpr(module->entry_func);
  return cp.const_map;
}


struct VMCompiler : ExprFunctor<void(const Expr& expr)> {
    std::vector<Instruction> instructions;
    std::unordered_map<Var, size_t, NodeHash, NodeEqual> var_map;
    size_t stack_index;
    bool seen_func;
    CompileEngine engine;
    VMCompilerContext* context;

    VMCompiler(VMCompilerContext* context) :
      instructions(), var_map(), stack_index(0),
      seen_func(false), engine(CompileEngine::Global()), context(context)  {}

    inline void Emit(const Instruction& instr) {
      CHECK((int)instr.op < 100) << "Invalid opcode " << (int)instr.op;
      switch (instr.op) {
        case Opcode::AllocDatatype:
        case Opcode::AllocTensor:
        case Opcode::GetField:
        case Opcode::Push:
        case Opcode::LoadConst:
          stack_index++;
          break;
        case Opcode::InvokePacked:
          stack_index -= instr.arity;
          break;
        case Opcode::If:
          stack_index--;
          break;
        case Opcode::Invoke:
          // this instruction will push one return value into stack
          stack_index++;
          break;
        case Opcode::Ret:
        case Opcode::Goto:
          break;
      }
      instructions.push_back(instr);
    }

    void VisitExpr_(const ConstantNode* const_node) {
      auto rconst = GetRef<Constant>(const_node);
      auto it = this->context->const_map.find(rconst);
      CHECK(it != this->context->const_map.end());
      Emit(LoadConst(it->second));
    }

    void VisitExpr_(const VarNode* var_node) {
      auto var = GetRef<Var>(var_node);
      auto it = this->var_map.find(var);
      CHECK(it != this->var_map.end());
      Emit(Push(it->second));
    }

    void VisitExpr_(const TupleNode* tuple_node) {
      auto tuple = GetRef<Tuple>(tuple_node);
      for (auto& field : tuple->fields) {
        this->VisitExpr(field);
      }
      // TODO: handle complex field expression
      // TODO: use correct tag
      Emit(AllocDatatype(0, tuple->fields.size()));
    }

    void VisitExpr_(const MatchNode* match_node) {
      auto match = GetRef<Match>(match_node);
      LOG(FATAL) << "translation of match nodes to the VM is"
                 << "is currently unsupported"
                 << std::endl;
    }

    void VisitExpr_(const LetNode* let_node) {
      this->VisitExpr(let_node->value);
      var_map.insert({ let_node->var, this->stack_index-1 });
      this->VisitExpr(let_node->body);
    }

    void VisitExpr_(const TupleGetItemNode* get_node) {
      auto get = GetRef<TupleGetItem>(get_node);
      this->VisitExpr(get->tuple);
      Emit(GetField(this->stack_index-1, get->index));
    }

    void VisitExpr_(const GlobalVarNode* gvar) {
      auto global = GetRef<GlobalVar>(gvar);
      auto it = this->context->global_map.find(global);
      CHECK(it != this->context->global_map.end());
      std::cout << "Invoke with: " << global->name_hint << "(func idx" << it->second << ")" << std::endl;
      Emit(Invoke(it->second));
    }

    void VisitExpr_(const IfNode* if_node) {
      this->VisitExpr(if_node->cond);
      auto after_cond = this->instructions.size();

      this->Emit(If(0, 0));

      auto stack_index_before_branch = stack_index;
      this->VisitExpr(if_node->true_branch);

      Emit(Goto(0));

      stack_index = stack_index_before_branch;
      auto after_true = this->instructions.size();
      this->VisitExpr(if_node->false_branch);

      auto after_false = this->instructions.size();

      // After we emit the true body, and false body,
      // we patch up the if instruction, and goto.
      auto true_offset = 1;
      auto false_offset = after_true - after_cond;
      this->instructions[after_cond].true_offset = true_offset;
      this->instructions[after_cond].false_offset = false_offset;
      // Patch the Goto.
      this->instructions[after_true - 1].pc_offset = after_false - after_true;
    }


    Instruction AllocTensorFromType(const TensorTypeNode* ttype) {
      std::vector<int64_t> shapes;
      for (auto sh : ttype->shape) {
        shapes.push_back(Downcast<tvm::Integer>(sh)->value);
      }
      DataType dtype = ttype->dtype;
      TVMType dltype = Type2TVMType(dtype);
      return AllocTensor(shapes, dltype);
    }

    void EmitInvokePrimitive(const Function& func, const Type& ret_type) {
      std::vector<Instruction> allocs;
      size_t return_num = 0;
      if (const TensorTypeNode* ttype = ret_type.as<TensorTypeNode>()) {
        // Allocate space for the return tensor.
        auto alloc = AllocTensorFromType(ttype);
        // Alloc to use for input
        allocs.push_back(alloc);
        // Alloc to use for output, but not used for this case. we should optmize that
        allocs.push_back(alloc);
        return_num = 1;
      } else if (const TupleTypeNode* ttype = ret_type.as<TupleTypeNode>()) {
        for (size_t i = 0; i < ttype->fields.size(); ++i) {
          auto f = ttype->fields[i];
          auto f_type = f.as<TensorTypeNode>();
          allocs.push_back(AllocTensorFromType(f_type));
        }
        return_num = ttype->fields.size();
        allocs.push_back(AllocDatatype(0, return_num));
      } else {
        LOG(FATAL) << "Unsupported return value type";
      }

      for (auto& alloc : allocs) {
        Emit(alloc);
      }

      // Next generate the invoke instruction.
      CHECK(func->IsPrimitive());
      auto target = Target::create("llvm");
      auto key = CCacheKeyNode::make(func, target);
      auto cfunc = engine->Lower(key);
      // TODO: support lowered funcs for multiple targets
      CHECK(cfunc->funcs.size() == 1);
      auto op_index = this->context->lowered_funcs.size();
      this->context->lowered_funcs.push_back(cfunc->funcs[0]);

      // If Tensor, 1
      // If Tuple, size of tuple
      size_t arity = func->params.size() + return_num;
      CHECK(arity < 10);
      Emit(InvokePacked(op_index, arity, return_num));
    }

    void VisitExpr_(const CallNode* call_node) {
      // First generate instructions to populate stack with arguments.
      std::vector<size_t> args;
      for (auto arg : call_node->args) {
        this->VisitExpr(arg);
        args.push_back(stack_index-1);
      }

      if ( !args.empty() && (args.back() - args.front()) != (args.size() - 1)) {
        std::cout << "Found non-consec sequence\n";
        for (auto& i : args) {
          Emit(Push(i));
        }
      }

      if (auto func_node = call_node->op.as<FunctionNode>()) {
        CHECK(func_node->IsPrimitive());
        EmitInvokePrimitive(
          GetRef<Function>(func_node),
          call_node->checked_type());
      } else if (auto global_node = call_node->op.as<GlobalVarNode>()) {
        auto global = GetRef<GlobalVar>(global_node);
        auto func = context->module->Lookup(global);
        auto it = this->context->global_map.find(global);
        CHECK(it != this->context->global_map.end());
        // If we are going to call a closure allocation
        // wrapper then we need to emit a closure here.
        if (IsClosure(func)) {
          Emit(AllocClosure(it->second, func->params.size()));
        } else {
          Emit(Invoke(it->second));
        }
      } else if (auto constructor_node = call_node->op.as<ConstructorNode>()) {
        auto constructor = GetRef<Constructor>(constructor_node);
        auto tag = GetConstructorTag(constructor);
        Emit(AllocDatatype(tag, call_node->args.size()));
      } else {
        LOG(FATAL) << "unsupported case in vm compiler: " << call_node->op;
      }
    }

    size_t GetConstructorTag(tvm::relay::Constructor constructor) {
      auto it = this->context->tag_map.find(constructor);
      if (it != this->context->tag_map.end()) {
        return it->second;
      } else {
        auto tag = this->context->tag_map.size();
        this->context->tag_map[constructor] = tag;
        this->context->tag_index_map[tag] = constructor;
        return tag;
      }
    }

    void VisitExpr_(const FunctionNode* func_node) {
      LOG(FATAL) << "local functions should have been removed by lambda lifting";
    }

    void CompileClosure(const Function& func) {
      // We will expect that the caller will properly
      // populate both the free variables and the arguments
      // on the stack.

      // We will first find the parameters to the outer
      // function (the free variables) on the stack.
      for (auto param : func->params) {
        var_map.insert({ param, this->stack_index++ });
      }

      // We will then layout the actual function arguments.
      auto inner_func = Downcast<Function>(func->body);
      for (auto param : inner_func->params) {
        var_map.insert({ param, this->stack_index++ });
      }

      // We will now process the body like normal.
      this->VisitExpr(inner_func->body);
    }

    void Compile(const Function& func) {
      RelayPrint(func, false);

      // We need to generate code specially for lifted closures.
      if (IsClosure(func)) {
        CompileClosure(func);
        return;
      }

      for (auto param : func->params) {
        std::cout << "Func param " << param << " at " << this->stack_index << "\n";
        var_map.insert({ param, this->stack_index++ });
      }

      this->VisitExpr(func->body);
    }
};

void PopulatePackedFuncMap(
  const std::vector<LoweredFunc>& lowered_funcs,
  std::vector<PackedFunc>* packed_funcs) {
  runtime::Module mod;
  if (lowered_funcs.size() > 0) {
    // TODO(@jroesch): we need to read target from build config
    Target target = Target::create("llvm");
    if (const auto* f = runtime::Registry::Get("relay.backend.build")) {
      mod = (*f)(tvm::Array<LoweredFunc>(lowered_funcs.begin(), lowered_funcs.end()), target);
    } else {
      LOG(FATAL) << "relay.backend.build is not registered";
    }
    CHECK(mod.operator->());
    for (auto lfunc : lowered_funcs) {
      packed_funcs->push_back(mod.GetFunction(lfunc->name));
    }
  }
}

VMFunction CompileFunc(VMCompilerContext* context, const GlobalVar& var, const Function& func) {
  size_t params = func->params.size();
  VMCompiler compiler(context);
  compiler.Compile(func);
  compiler.instructions.push_back(Ret());
  // Would like to refactor this so we only check if closure once.
  if (IsClosure(func)) {
    return VMFunction(var->name_hint, params, compiler.instructions);
  } else {
    auto inner_params = Downcast<Function>(func->body)->params.size();
    return VMFunction(var->name_hint, params + inner_params, compiler.instructions);
  }
}

Module OptimizeModule(const Module& mod) {
  return LambdaLift(mod);
}

void PopulateGlobalMap(GlobalMap* global_map, const Module& mod) {
  // First we populate global map.
  size_t global_index = 0;
  for (auto named_func : mod->functions) {
    auto gvar = named_func.first;
    global_map->insert({ gvar, global_index++ });
  }
}

VirtualMachine CompileModule(const Module& mod_ref) {
  Module mod = mod_ref;
  // Run some optimizations first, this code should
  // be moved to pass manager.

  mod = OptimizeModule(mod);

  VirtualMachine vm;

  VMCompilerContext context;
  context.module = mod;

  // Populate the global map.
  //
  // This maps global variables to a global index
  // in the VMFunction table.
  PopulateGlobalMap(&context.global_map, mod);

  // Next we populate constant map.
  context.const_map = LayoutConstantPool(mod);

  // Next we get ready by allocating space for
  // the global state.
  vm.functions.resize(mod->functions.size());
  vm.constants.resize(context.const_map.size());

  for (auto pair : context.const_map) {
    vm.constants[pair.second] = VMTensor(pair.first->data);
  }

  for (auto named_func : mod->functions) {
    auto gvar = named_func.first;
    auto func = named_func.second;
    auto vm_func = CompileFunc(&context, gvar, func);

    size_t func_index = context.global_map.at(gvar);
    CHECK(func_index < vm.functions.size());
    vm.functions[func_index] = vm_func;
  }

  for (auto vm_func : vm.functions) {
    std::cout << "Function: " << vm_func.name
      << std::endl
      << vm_func
      << "-------------" << std::endl;
  }

  PopulatePackedFuncMap(context.lowered_funcs, &vm.packed_funcs);

  vm.global_map = context.global_map;
  vm.tag_index_map = context.tag_index_map;

  return vm;
}

}  // namespace vm
}  // namespace relay
}  // namespace tvm
