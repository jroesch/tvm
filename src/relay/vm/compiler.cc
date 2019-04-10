/*!
 *  Copyright (c) 2018 by Contributors
 * \file tvm/relay/runtime/runtime.h
 * \brief Abstract device memory management API
 */

#include <tvm/runtime/memory_manager.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/vm/vm.h>
#include <tvm/relay/error.h>
#include <tvm/relay/interpreter.h>
#include <tvm/relay/logging.h>
#include <tvm/relay/pass.h>
#include "../backend/compile_engine.h"
#include "../../runtime/naive_allocator.h"

#include <vector>
#include <iostream>
#include <unordered_map>
#include <unordered_set>

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
  ErrorReporter err_reporter;
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


void InstructionPrint(std::ostream& os, const Instruction& instr);

struct VMCompiler : ExprFunctor<void(const Expr& expr)> {
    /*! \brief Store the expression a variable points to. */
    std::unordered_map<Var, Expr, NodeHash, NodeEqual> expr_map;

    std::vector<Instruction> instructions;

    // var -> register num
    std::unordered_map<Var, VirtualRegisterNum, NodeHash, NodeEqual> var_register_map;

    size_t last_register;

    // Total number of virtual registers allocated
    size_t registers_num;
    CompileEngine engine;

    /*! \brief The functions that have been lowered. */
    std::unordered_map<LoweredFunc, size_t, NodeHash, NodeEqual> seen_funcs;

    /*! \brief Global shared meta data */
    VMCompilerContext* context;

    VMCompiler(VMCompilerContext* context) :
      instructions(), last_register(0), registers_num(1),
      engine(CompileEngine::Global()), context(context)  {}

    size_t NewRegister() {
      return registers_num++;
    }
    
    inline void Emit(const Instruction& instr) {
      RELAY_LOG(INFO) << "VMCompiler::Emit: instr=" << instr;
      CHECK((int)instr.op < 100) << "Invalid opcode " << (int)instr.op;
      switch (instr.op) {
        case Opcode::AllocDatatype:
        case Opcode::AllocTensor:
        case Opcode::GetField:
        case Opcode::LoadConst:
        case Opcode::Select:
        case Opcode::Invoke:
        case Opcode::AllocClosure:
        case Opcode::Move:
          last_register = instr.dst;
          break;
        case Opcode::InvokePacked:
          last_register = instr.packed_args[instr.arity - 1];
          break;
        case Opcode::InvokeClosure:
        case Opcode::If:
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
      Emit(LoadConst(it->second, NewRegister()));
    }

    void VisitExpr_(const VarNode* var_node) {
      auto var = GetRef<Var>(var_node);
      auto reg_it = this->var_register_map.find(var);
      CHECK(reg_it != this->var_register_map.end());
      last_register = reg_it->second;
    }

    void VisitExpr_(const TupleNode* tuple_node) {
      auto tuple = GetRef<Tuple>(tuple_node);
      std::vector<size_t> fields_registers;

      for (auto& field : tuple->fields) {
        this->VisitExpr(field);
        fields_registers.push_back(last_register);
      }

      // TODO: handle complex field expression
      // TODO: use correct tag
      Emit(AllocDatatype(0, tuple->fields.size(), fields_registers, NewRegister()));
    }

    void VisitExpr_(const MatchNode* match_node) {
      auto match = GetRef<Match>(match_node);
      LOG(FATAL) << "translation of match nodes to the VM is"
                 << "currently unsupported"
                 << std::endl;
    }

    void VisitExpr_(const LetNode* let_node) {
      // Let binding value will be the last value pushed,
      // when we visit the value it will generate a sequence
      // of instructions that will leave the final value
      // on the stack.
      //
      RELAY_LOG(INFO) << let_node->value << std::endl;
      this->VisitExpr(let_node->value);
      RELAY_LOG(INFO) << this->last_register << std::endl;      
      var_register_map.insert({ let_node->var, this->last_register });      
      this->VisitExpr(let_node->body);
    }

    void VisitExpr_(const TupleGetItemNode* get_node) {
      auto get = GetRef<TupleGetItem>(get_node);
      this->VisitExpr(get->tuple);
      auto tuple_register = last_register;
      Emit(GetField(tuple_register, get->index, NewRegister()));
    }

    void VisitExpr_(const GlobalVarNode* gvar) {
      LOG(FATAL) << "Global variables should only appear in the call position";
    }

    void VisitExpr_(const IfNode* if_node) {
      this->VisitExpr(if_node->cond);

      size_t cond_register = last_register;

      auto after_cond = this->instructions.size();

      this->Emit(If(cond_register, 0, 0));
      this->VisitExpr(if_node->true_branch);

      size_t true_register = last_register;

      Emit(Goto(0));

      // Finally store how many instructions there are in the
      // true branch.
      auto after_true = this->instructions.size();

      this->VisitExpr(if_node->false_branch);

      size_t false_register = last_register;

      // Compute the total number of instructions
      // after generating false.
      auto after_false = this->instructions.size();

      // Now we will compute the jump targets in order
      // to properly patch the instruction with the
      // the requiste targets.

      // After we emit the true body, and false body,
      // we patch up the if instruction, and goto.
      auto true_offset = 1;
      auto false_offset = after_true - after_cond;
      this->instructions[after_cond].true_offset = true_offset;
      this->instructions[after_cond].false_offset = false_offset;

      // Patch the Goto.
      // CHECK(this->instructions[after_true - 1].op == Opcode::Goto);
      this->instructions[after_true - 1].pc_offset = (after_false - after_true) + 1;

      Emit(Select(cond_register, true_register, false_register, NewRegister()));
    }

    Instruction AllocTensorFromType(const TensorTypeNode* ttype) {
      std::vector<int64_t> shapes;
      for (auto sh : ttype->shape) {
        shapes.push_back(Downcast<tvm::Integer>(sh)->value);
      }
      DataType dtype = ttype->dtype;
      TVMType dltype = Type2TVMType(dtype);
      return AllocTensor(shapes, dltype, NewRegister());
    }

    void EmitInvokePrimitive(const Function& func, std::vector<size_t> args_registers, const Type& ret_type) {
      std::vector<Instruction> allocs;
      size_t return_num = 0;
      if (const TensorTypeNode* ttype = ret_type.as<TensorTypeNode>()) {
        // Allocate space for the return tensor.
        auto alloc = AllocTensorFromType(ttype);
        // // Alloc to use for input
        // allocs.push_back(alloc);
        // Alloc to use for output, but not used for this case. we should optmize that
        allocs.push_back(alloc);
        return_num = 1;
      } else if (const TupleTypeNode* ttype = ret_type.as<TupleTypeNode>()) {
        std::vector<size_t> fields_registers;

        for (size_t i = 0; i < ttype->fields.size(); ++i) {
          auto f = ttype->fields[i];
          auto f_type = f.as<TensorTypeNode>();
          allocs.push_back(AllocTensorFromType(f_type));
          fields_registers.push_back(allocs.back().dst);
        }
        return_num = ttype->fields.size();
      } else {
        LOG(FATAL) << "Unsupported return value type";
      }

      for (auto& alloc : allocs) {
        Emit(alloc);
        args_registers.push_back(alloc.dst);
      }

      // Next generate the invoke instruction.
      CHECK(func->IsPrimitive());
      auto target = Target::create("llvm");
      auto key = CCacheKeyNode::make(func, target);
      auto cfunc = engine->Lower(key);
      // TODO: support lowered funcs for multiple targets
      CHECK(cfunc->funcs.size() == 1);
      auto op_index = -1;
      if (seen_funcs.find(cfunc->funcs[0]) == seen_funcs.end()) {
        op_index = this->context->lowered_funcs.size();
        this->context->lowered_funcs.push_back(cfunc->funcs[0]);
        seen_funcs[cfunc->funcs[0]] = op_index;
      } else {
        op_index = seen_funcs[cfunc->funcs[0]];
      }

      // If Tensor, 1
      // If Tuple, size of tuple
      size_t arity = func->params.size() + return_num;
      Emit(InvokePacked(op_index, arity, return_num, args_registers));
      if (return_num > 1) {
        // return value is a tuple, we need to create a tuple
        std::vector<size_t> fields_registers;
        for (size_t i = func->params.size(); i < arity; ++i) {
          fields_registers.push_back(args_registers[i]);
        }
        Emit(AllocDatatype(0, return_num, fields_registers, NewRegister()));
      }
    }

    void VisitExpr_(const CallNode* call_node) {
      // First generate instructions to populate stack with arguments.

      std::vector<size_t> args_registers;

      for (auto arg : call_node->args) {
        CHECK(arg.as<VarNode>())
          << "found: " << RelayPrint(arg, false);
        this->VisitExpr(arg);
        args_registers.push_back(last_register);
      }

      Expr op = call_node->op;

      if (auto func_node = op.as<FunctionNode>()) {
        CHECK(func_node->IsPrimitive());
        EmitInvokePrimitive(
          GetRef<Function>(func_node),
          args_registers,
          call_node->checked_type());
      } else if (auto global_node = op.as<GlobalVarNode>()) {
        auto global = GetRef<GlobalVar>(global_node);
        auto it = this->context->global_map.find(global);
        CHECK(it != this->context->global_map.end());
        RELAY_LOG(INFO)
          << "VisitExpr_: generating invoke for "
          << global->name_hint
          << " with func_index="
          << it->second;

        auto func = this->context->module->Lookup(global);
        if (IsClosure(func)) {
          auto arity = func->params.size();
          std::vector<size_t> free_var_registers;
          for (size_t i = 0; i < arity; ++i) {
            free_var_registers.push_back(var_register_map.at(func->params[i]));
          }
          Emit(AllocClosure(it->second, arity, free_var_registers, NewRegister()));
        } else {
          Emit(Invoke(it->second, args_registers, NewRegister()));
          // 0 is return value register
          Emit(Move(0, NewRegister()));
        }
      } else if (auto constructor_node = op.as<ConstructorNode>()) {
        auto constructor = GetRef<Constructor>(constructor_node);
        auto tag = GetConstructorTag(constructor);
        Emit(AllocDatatype(tag, call_node->args.size(), args_registers, NewRegister()));
      } else if (auto var_node = op.as<VarNode>()) {
        VisitExpr(GetRef<Var>(var_node));
        Emit(InvokeClosure(last_register, args_registers, NewRegister()));
        Emit(Move(0, NewRegister()));
      } else {
        LOG(FATAL) << "unsupported case in vm compiler: " << op;
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
      if (!func_node->IsPrimitive()) {
        LOG(FATAL) << "local functions should have been removed by lambda lifting"
                  << RelayPrint(GetRef<Function>(func_node), false);
      }
    }

    void CompileClosure(const Function& func) {
      // We first layout the function arguments.
      auto inner_func = Downcast<Function>(func->body);

      size_t i = 1;
      for (auto param : inner_func->params) {
        auto arg_register = NewRegister();
        CHECK_EQ(i, arg_register);
        var_register_map.insert({ param, arg_register });
        i++;
      }

      // We then assign register num to the free variables
      for (auto param : func->params) {
        auto arg_register = NewRegister();        
        CHECK_EQ(i, arg_register);        
        var_register_map.insert({ param, arg_register });        
        i++;
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

      for (auto i = 0; i < func->params.size(); ++i) {
        auto arg_register = NewRegister();
        CHECK_EQ(arg_register, i+1);
        var_register_map.insert({ func->params[i], arg_register });
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
  RELAY_LOG(INFO) << "CompileFunc: " << std::endl
    << RelayPrint(func, false) << std::endl;
  size_t params = func->params.size();
  VMCompiler compiler(context);
  compiler.Compile(func);
  // return the last evaluated expression
  compiler.instructions.push_back(Ret(compiler.last_register));

  // Would like to refactor this so we only check if closure once.
  if (IsClosure(func)) {
    auto inner_params = Downcast<Function>(func->body)->params.size();
    return VMFunction(var->name_hint, params + inner_params, compiler.instructions, compiler.registers_num);
  } else {
    return VMFunction(var->name_hint, params, compiler.instructions, compiler.registers_num);
  }
}

Module OptimizeModule(const Module& mod) {
  ToANormalForm(mod->entry_func, mod);
  InlinePrimitives(mod);
  LambdaLift(mod);
  return InlinePrimitives(mod);
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
    vm.constants[pair.second] = TensorObj(pair.first->data);
  }

  for (auto named_func : mod->functions) {
    auto gvar = named_func.first;
    auto func = named_func.second;
    auto vm_func = CompileFunc(&context, gvar, func);

    size_t func_index = context.global_map.at(gvar);
    CHECK(func_index < vm.functions.size());
    vm.functions[func_index] = vm_func;
  }

#ifdef USE_RELAY_LOG
  for (auto vm_func : vm.functions) {
    std::cout << "Function: " << vm_func.name
      << std::endl
      << vm_func
      << "-------------" << std::endl;
  }
#endif  // USE_RELAY_LOG

  PopulatePackedFuncMap(context.lowered_funcs, &vm.packed_funcs);

  vm.global_map = context.global_map;
  vm.tag_index_map = context.tag_index_map;

  return vm;
}

}  // namespace vm
}  // namespace relay
}  // namespace tvm
