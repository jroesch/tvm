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

// Packed Function extensions.
TVMRetValue& runtime::TVMRetValue::operator=(relay::vm::VMObject other) {
  this->SwitchToClass(kVMObject, other);
  return *this;
}

runtime::TVMArgValue::operator relay::vm::VMObject() const {
  if (type_code_ == kNull) return relay::vm::VMObject(nullptr);
  TVM_CHECK_TYPE_CODE(type_code_, kVMObject);
  return *ptr<relay::vm::VMObject>();
}

namespace relay {
namespace vm {


Instruction::Instruction() {}

Instruction::Instruction(const Instruction& instr) {
  this->op = instr.op;
  switch (instr.op) {
    case Opcode::Push:
      this->stack_index = instr.stack_index;
      return;
    case Opcode::Ret:
      return;
    case Opcode::AllocTensor:
      this->tensor_info = instr.tensor_info;
      return;
    case Opcode::AllocDatatype:
      this->constructor_tag = instr.constructor_tag;
      this->num_fields = instr.num_fields;
      return;
    case Opcode::InvokePacked:
      this->packed_index = instr.packed_index;
      this->arity = instr.arity;
      this->output_size = instr.output_size;
      return;
    case Opcode::If:
      this->true_offset = instr.true_offset;
      this->false_offset = instr.false_offset;
      return;
    case Opcode::Invoke:
      this->func_index = instr.func_index;
      return;
    case Opcode::LoadConst:
      this->const_index = instr.const_index;
      return;
    case Opcode::GetField:
      this->object_offset = instr.object_offset;
      this->field_index = instr.field_index;
      return;
    case Opcode::Goto:
      this->pc_offset = instr.pc_offset;
      return;
  }
}

// TODO(@jroesch): this leaks memory fix me
Instruction::~Instruction() {}

Instruction Push(size_t stack_index) {
  Instruction instr;
  instr.op = Opcode::Push;
  instr.stack_index = stack_index;
  return instr;
}

Instruction Ret() {
  Instruction instr;
  instr.op = Opcode::Ret;
  return instr;
}

Instruction InvokePacked(size_t packed_index, size_t arity, size_t output_size) {
  Instruction instr;
  instr.op = Opcode::InvokePacked;
  instr.packed_index = packed_index;
  instr.arity = arity;
  instr.output_size = output_size;
  return instr;
}

Instruction AllocTensor(const std::vector<int64_t> shape, DLDataType dtype) {
  Instruction instr;
  instr.op = Opcode::AllocTensor;
  instr.tensor_info.shape = new int64_t[shape.size()];
  instr.tensor_info.ndim = shape.size();
  std::memcpy(
      reinterpret_cast<void*>(instr.tensor_info.shape),
      reinterpret_cast<const void*>(shape.data()),
      shape.size() * sizeof(int64_t));
  instr.tensor_info.dtype = dtype;
  return instr;
}

Instruction AllocDatatype(size_t tag, size_t num_fields) {
  Instruction instr;
  instr.op = Opcode::AllocDatatype;
  instr.constructor_tag = tag;
  instr.num_fields = num_fields;
  return instr;
}

Instruction GetField(size_t object_offset, size_t field_index) {
  Instruction instr;
  instr.op = Opcode::GetField;
  instr.object_offset = object_offset;
  instr.field_index = field_index;
  return instr;
}

Instruction If(size_t true_branch, size_t false_branch) {
  Instruction instr;
  instr.op = Opcode::If;
  instr.true_offset = true_branch;
  instr.false_offset = false_branch;
  return instr;
}

Instruction Goto(size_t pc_offset) {
  Instruction instr;
  instr.op = Opcode::Goto;
  instr.pc_offset = pc_offset;
  return instr;
}

Instruction Invoke(size_t func_index) {
  Instruction instr;
  instr.op = Opcode::Invoke;
  instr.func_index = func_index;
  return instr;
}

Instruction LoadConst(size_t const_index) {
  Instruction instr;
  instr.op = Opcode::LoadConst;
  instr.const_index = const_index;
  return instr;
}

void InstructionPrint(std::ostream& os, const Instruction& instr) {
  switch (instr.op) {
    case Opcode::Push: {
      os << "push " << instr.stack_index;
      break;
    }
    case Opcode::Ret: {
      os << "ret";
      break;
    }
    case Opcode::InvokePacked: {
      os << "invoke_packed ";
      os << instr.packed_index;
      os << " " << instr.arity;
      os << " " << instr.output_size;
      break;
    }
    case Opcode::AllocTensor: {
      os << "alloc_tensor";
      os << "(";

      for (size_t i = 0; i < instr.tensor_info.ndim; i++) {
        os << instr.tensor_info.shape[i] << ", ";
      }
      os << ") ";
      os << TVMType2Type(instr.tensor_info.dtype);
      break;
    }
    case Opcode::AllocDatatype: {
      os << "alloc_block";
      os << " ";
      os << instr.constructor_tag << " ";
      os << instr.num_fields;
      break;
    }
    case Opcode::If: {
      os << "if "
         << instr.true_offset << " "
         << instr.false_offset;
      break;
    }
    case Opcode::Invoke: {
      os << "invoke "
         << instr.func_index;
      break;
    }
    case Opcode::LoadConst: {
      os << "load_const "
         << instr.const_index;
      break;
    }
    case Opcode::GetField: {
      os << "get_field "
         << instr.object_offset << " "
         << instr.field_index;
      break;
    }
    case Opcode::Goto: {
      os << "goto "
         << instr.pc_offset;
      break;
    }
    default:
      os << "unknown instruction " << (int)instr.op;
      break;
  }
}

void VMFunctionPrint(const VMFunction& vm_func) {
  return;
  std::cout << vm_func.name << ": " << std::endl;
  for (size_t i = 0; i < vm_func.instructions.size(); ++i) {
    std::cout << i << ": ";
    InstructionPrint(std::cout, vm_func.instructions[i]);
    std::cout << ";" << std::endl;
  }
}

using TagMap = std::unordered_map<tvm::relay::Constructor, size_t, NodeHash, NodeEqual>;
using TagNameMap = std::unordered_map<size_t, tvm::relay::Constructor>;
using GlobalMap = std::unordered_map<GlobalVar, size_t, NodeHash, NodeEqual>;
using ConstMap = std::unordered_map<Constant, size_t, NodeHash, NodeEqual>;

struct VMCompilerContext {
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

    /*! @brief local variable's relative position(to bp) in stack */
    std::unordered_map<Var, size_t, NodeHash, NodeEqual> var_map;

    /*!
     * @brief the next place available in stack to store value, start from 0
     * for each function
     */
    size_t stack_index;

    bool seen_func;
    CompileEngine engine;

    /*! @brief global shared meta data */
    VMCompilerContext* context;

    VMCompiler(VMCompilerContext* context) :
      instructions(), var_map(), stack_index(0),
      seen_func(false), engine(CompileEngine::Global()), context(context)  {}

    /*!
     * @brief Emit Opcode and adjust stack_index accordingly
     * 
     * @param instr Instruction to emit
     */
    inline void Emit(const Instruction& instr) {
      CHECK((int)instr.op < 100) << "Invalid opcode " << (int)instr.op;
      switch (instr.op) {
        case Opcode::AllocDatatype:
        case Opcode::AllocTensor:
        case Opcode::GetField:
        case Opcode::Push:
        case Opcode::LoadConst:
          // these intructions will push one value to stack
          stack_index++;
          break;
        case Opcode::InvokePacked:
          // this instruction will pop instr.arity value from stack
          stack_index -= instr.arity;
          break;
        case Opcode::If:
          // this instruction will pop one value from stack
          stack_index--;
          break;
        default:
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
      std::cout << "Ignore compiling match node\n";
    }
  
    void VisitExpr_(const LetNode* let_node) {
      this->VisitExpr(let_node->value);
      // let binding value will be at stack_index-1
      var_map.insert({ let_node->var, this->stack_index-1 });
      this->VisitExpr(let_node->body);
    }

    void VisitExpr_(const TupleGetItemNode* get_node) {
      auto get = GetRef<TupleGetItem>(get_node);
      this->VisitExpr(get->tuple);
      // tuple value will be at stack_index-1
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

      // save the stack_index before entering true branch
      auto stack_index_before_branch = stack_index;
      this->VisitExpr(if_node->true_branch);

      Emit(Goto(0));

      // restore stack_index to the value before the branch
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

      // some arguments might be composite expressions, not atomic variable, we
      // need to push the arguments again to keep the invariant that the arguments
      // of function are consecutive to each other
      if (!args.empty() && (args.back() - args.front()) != (args.size() - 1)) {
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
        auto it = this->context->global_map.find(global);
        CHECK(it != this->context->global_map.end());
        Emit(Invoke(it->second));
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
      CHECK(!seen_func) << GetRef<Function>(func_node);
      this->seen_func = true;
      for (auto param : func_node->params) {
        std::cout << "Func param " << param << " at " << this->stack_index << "\n";
        var_map.insert({ param, this->stack_index++ });
      }

      this->VisitExpr(func_node->body);
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
//  std::cout << func << std::endl;
  size_t params = func->params.size();
  VMCompiler compiler(context);
  // std::cout << "Compiling: " << func << std::endl;
  std::cout << "Start: " << compiler.stack_index << "\n";
  compiler.VisitExpr(func);
  compiler.instructions.push_back(Ret());
  VMFunction vm_func = VMFunction(var->name_hint, params, compiler.instructions);
  return vm_func;
}

VirtualMachine CompileModule(const Module& mod) {
  VirtualMachine vm;

  VMCompilerContext context;


  // First we populate global map.
  size_t global_index = 0;
  for (auto named_func : mod->functions) {
    auto gvar = named_func.first;
    context.global_map.insert({ gvar, global_index++ });
    std::cout << "Global Map " << gvar << " " << global_index-1 << std::endl;
  }

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
    std::cout << "Compiling func " << gvar->name_hint << std::endl;
    auto vm_func = CompileFunc(&context, gvar, func);

    size_t func_index = context.global_map.at(gvar);
    CHECK(func_index < vm.functions.size());
    vm.functions[func_index] = vm_func;
  }

  for (auto vm_func : vm.functions) {
    std::cout << "Function: " << vm_func.name << std::endl;
    VMFunctionPrint(vm_func);
    std::cout << "-------------" << std::endl;    
  }

  PopulatePackedFuncMap(context.lowered_funcs, &vm.packed_funcs);

  vm.global_map = context.global_map;
  vm.tag_index_map = context.tag_index_map;

  return vm;
}

void VirtualMachine::PushFrame(size_t arg_count, size_t ret_pc, size_t sp, const VMFunction& vm_func) {
  DumpStack();
  auto frame = VMFrame(ret_pc, bp, sp, func_index, arg_count, code);
  frames.push_back(frame);
}

size_t VirtualMachine::PopFrame() {
  CHECK(frames.size() != 0);
  const VMFrame& fr = frames.back();
  auto stack_size = stack.size();
  // Copy return value;
  CHECK(stack_size - fr.args - 1 < stack.size())
    << "attempting to read stack slot: "
    << stack_size - fr.args - 1
    << " stack_size: "
    << stack_size;

  CHECK(0 <= stack_size - fr.sp);
  // cope return value to the position past last function's frame
  stack[fr.sp] = stack[stack_size - 1];
  // Resize value stack.
  stack.resize(fr.sp + 1);
  // Reset frame.
  std::cout << "Reset frame " << bp << " -> " << fr.bp << "\n";
  std::cout << "Reset stack " << stack_size << " -> " << stack.size() << "\n";
  bp = fr.bp;
  pc = fr.pc;
  func_index = fr.func_index;
  code = fr.code;
  auto call_stack_size = frames.size();
  frames.pop_back();
  return call_stack_size;
}

void VirtualMachine::InvokeGlobal(const VMFunction& func, const std::vector<VMObject>& args) {
  std::cout << "===================\nInvoking global " << func.name << std::endl;
  for (auto arg : args) {
    stack.push_back(arg);
  }

  PushFrame(func.params, this->pc + 1, stack.size(), func);
  std::cout << "func.params= " << func.params << ", stack.size()= " << stack.size() << std::endl;

  code = func.instructions.data();
  pc = 0;
  bp = stack.size() - func.params;
}

VMObject VirtualMachine::Invoke(const VMFunction& func, const std::vector<VMObject>& args) {
  std::cout << "Executing function " << func.name << " bp " << bp << std::endl;
  InvokeGlobal(func, args);
  Run();
  auto alloc = MemoryManager::Global()->GetAllocator(ctxs[0]);
  std::cout << "Memory used: " << alloc->UsedMemory() << " B\n";
  // std::cout << "final stack size: " << stack.size() << "bp: " << bp << std::endl;
  return stack.back();
}

VMObject VirtualMachine::Invoke(const GlobalVar& global, const std::vector<VMObject>& args) {
  auto func_index = this->global_map[global];
  std::cout << "Invoke Global " << global << " at index " << func_index << std::endl;
  return Invoke(this->functions[func_index], args);
}

void InvokePacked(const PackedFunc& func, size_t arg_count, size_t output_size, std::vector<VMObject>& stack) {
  CHECK(arg_count <= stack.size());

  std::vector<TVMValue> values(arg_count);
  std::vector<int> codes(arg_count);
  runtime::TVMArgsSetter setter(values.data(), codes.data());

  auto stack_start = stack.size() - arg_count - 1;
  for (size_t i = 0; i < arg_count; i++) {
    NDArray data = ToNDArray(stack[stack_start + i]);
    setter(i, data);
  }

  TVMRetValue rv;
  func.CallPacked(TVMArgs(values.data(), codes.data(), arg_count), &rv);

  // Fix the object at return value position
  if (output_size == 1) {
    stack[stack.size() - 1] = stack[stack.size() - 2];
  } else {
    auto adt = std::dynamic_pointer_cast<VMDatatypeCell>(stack.back().ptr);
    for (size_t i = 0; i < output_size; ++i) {
      adt->fields[i] = stack[stack.size() - output_size - 1 + i];
    }
  }

  // We can do this more efficiently by reverse laying out the arguments
  // and just shrinking the stack.
  stack[stack.size() - arg_count - 1] = stack[stack.size() - 1];
  stack.resize(stack.size() - arg_count);
}

void VirtualMachine::Init(const std::vector<TVMContext>& ctxs) {
  this->ctxs = ctxs;
}

void VirtualMachine::DumpRegister() {
  if (!this->debug) {
    return;
  }
  std::cout << std::endl << "-- Registers: --\n";
  std::cout << "Bp: " << bp << std::endl;
  std::cout << "Stack Size: " << stack.size() << std::endl;
  std::cout << "Frame Size: " << frames.size() << std::endl;
  std::cout << "----\n" ;
}

void VirtualMachine::DumpStack() {
  if (!this->debug) {
    return;
  }
  std::cout << "DumpStack---\n";
  for (size_t i = bp; i < stack.size(); ++i) {
    std::cout << i << " " << (int)stack[i]->tag << " ";
    switch (stack[i]->tag) {
      case VMObjectTag::kTensor: {
        VMTensorCell* tensor = (VMTensorCell*)stack[i].operator->();
        std::cout << tensor->data->ndim;
        if (tensor->data->ndim == 0) {
          std::cout << " " << *((int*)(tensor->data->data));
        }
        std::cout << " \n";
        break;
      }
      default: {
        std::cout << "\n";
      }
    }
  }
  std::cout << "DumpStack end---\n";
}

void VirtualMachine::Run() {
  CHECK(this->code);
  this->pc = 0;
  auto stack_start = frames.size();
  while (true) {
  main_loop:
    auto const& instr = this->code[this->pc];
    std::cout << "Executing(" << pc << "): " ;
    InstructionPrint(std::cout, instr);
    std::cout << "\n";
    DumpRegister();

    switch (instr.op) {
      case Opcode::LoadConst: {
        stack.push_back(this->constants[instr.const_index]);
        pc++;
        goto main_loop;
      }
      case Opcode::Invoke: {
        // VMFunctionPrint(this->functions[this->func_index]);
        InvokeGlobal(this->functions[instr.func_index], {});
        goto main_loop;
      }
      case Opcode::InvokePacked: {
        auto start_stack = stack.size();
        const auto& func = packed_funcs[instr.packed_index];
        const auto& arity = instr.arity;
        // std::cout << "before call" << std::endl;
        // std::cout << this->stack.size() << std::endl;
        DumpStack();
        InvokePacked(func, arity, instr.output_size, stack);
        DumpStack();        
        CHECK(start_stack - arity == stack.size())
          << "start_stack: " << start_stack
          << "end_stack: " << stack.size();
        // std::cout << "after call" << std::endl;
        pc++;
        goto main_loop;
      }
      case Opcode::GetField: {
        auto object = stack[bp + instr.object_offset];
        DumpStack();
        CHECK(object->tag == VMObjectTag::kDatatype) << "Object is not data type object " << bp << " " << instr.object_offset << " " << (int)object->tag;
        const std::shared_ptr<VMDatatypeCell>& tuple = std::dynamic_pointer_cast<VMDatatypeCell>(object.ptr);
        auto field = tuple->fields[instr.field_index];
        stack.push_back(field);
        pc++;
        goto main_loop;
      }
      case Opcode::Goto: {
        pc += instr.pc_offset + 1;
        goto main_loop;
      }
      case Opcode::If: {
        DumpStack();
        // How do we do this efficiently?
        DLContext cpu_ctx;
        cpu_ctx.device_type = kDLCPU;
        cpu_ctx.device_id = 0;

        const auto& cond = stack.back();
        NDArray cpu_array = ToNDArray(cond).CopyTo(cpu_ctx);
        CHECK_EQ(TVMType2Type(cpu_array->dtype), Bool());
        bool branch = reinterpret_cast<uint8_t*>(cpu_array->data)[0];

        // Remove cond.
        stack.pop_back();

        if (branch) {
          pc += instr.true_offset;
        } else {
          pc += instr.false_offset;
        }

        goto main_loop;
      }
      case Opcode::AllocTensor: {
        const auto& ti = instr.tensor_info;
        auto shape = std::vector<int64_t>(ti.ndim);
        shape.assign(ti.shape, ti.shape + ti.ndim);
        auto allocator = MemoryManager::Global()->GetAllocator(ctxs[0]);
        auto data = NDArray::Empty(shape, ti.dtype, ctxs[0], allocator);
        stack.push_back(VMTensor(data));
        pc++;
        goto main_loop;
      }
      case Opcode::AllocDatatype: {
        std::vector<VMObject> fields;
        size_t stack_size = stack.size();
        for (size_t i = 0; i < instr.num_fields; ++i) {
          fields.push_back(stack[stack_size - instr.num_fields + i]);
        }
        stack.push_back(VMDatatype(instr.constructor_tag, fields));
        pc++;
        goto main_loop;
      }
      case Opcode::Push: {
        CHECK(bp + instr.stack_index < stack.size()) << bp << " " << instr.stack_index << " " << stack.size();
        stack.push_back(stack[bp + instr.stack_index]);
        DumpStack();        
        pc++;
        goto main_loop;
      }
      case Opcode::Ret: {
        // If we have hit the point from which we started
        // running, we should return to the caller breaking
        // the dispatch loop.
        if (PopFrame() == stack_start) {
          return;
        // Otherwise we are just returning from a local call.
        //
        // Since we have already popped the stack we will just
        // resume at the top of the dispatch loop.
        } else {
          goto main_loop;
        }
      }
    }
  }
}

VirtualMachine VirtualMachine::FromModule(const Module& module,
                                          const std::vector<TVMContext>& ctxs) {
  auto vm = CompileModule(module);
  vm.Init(ctxs);
  return vm;
}

/*! \brief Convert from an array of relay.Value into VM compatible objects.
 */
void ConvertArgsToVM(tvm::Array<Value> args, std::vector<VMObject>& out) {
  for (auto arg : args) {
    if (auto tensor = arg.as<TensorValueNode>()) {
      out.push_back(VMTensor(tensor->data));
    } else if (auto tuple = arg.as<TupleValueNode>()) {
      std::vector<VMObject> fields;
      for (auto field : tuple->fields) {
        ConvertArgsToVM({field}, fields);
      }
      out.push_back(VMDatatype(0, fields));
    } else {
      LOG(FATAL) << "unknown case: " << arg;
    }
  }
}

/*! \brief Convert from an array of relay.Value into VM compatible objects.
 */
VMObject ValueToVM(Value value) {
  std::vector<VMObject> out;
  ConvertArgsToVM({value}, out);
  CHECK_LT(out.size(), 2);
  return out[0];
}

Value VMToValue(TagNameMap& tag_index_map, VMObject obj) {
  switch (obj->tag) {
    case VMObjectTag::kTensor: {
      return TensorValueNode::make(ToNDArray(obj));
    }
    case VMObjectTag::kDatatype: {
      auto data_type = std::dynamic_pointer_cast<VMDatatypeCell>(obj.ptr);

      tvm::Array<Value> fields;
      for (size_t i = 0; i < data_type->fields.size(); ++i) {
        fields.push_back(VMToValue(tag_index_map, data_type->fields[i]));
      }

      return ConValueNode::make(tag_index_map[data_type->tag], fields);
    }
    default:
      LOG(FATAL) << "unsupported return value";
      return Value();
  }
}

std::tuple<VMObject, TagNameMap> 
EvaluateModule(const Module& module, const std::vector<TVMContext> ctxs,
               const std::vector<VMObject>& vm_args) {
  VirtualMachine vm = VirtualMachine::FromModule(module, ctxs);
  std::cout << "Entry function is " << module->entry_func << std::endl;
  return std::make_tuple(vm.Invoke(module->entry_func, vm_args), vm.tag_index_map);
}

TVM_REGISTER_API("relay._vm._ValueToVM")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    *ret = ValueToVM(args[0]);
});

TVM_REGISTER_API("relay._vm._VMToValue")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    TagNameMap tag_index_map{};
    *ret = VMToValue(tag_index_map, args[0]);
});

TVM_REGISTER_API("relay._vm._Tensor")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    *ret = VMTensor(args[0]);
});

TVM_REGISTER_API("relay._vm._Tuple")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  std::vector<VMObject> fields;
  for (size_t i = 0; i < args.size(); i++) {
    fields.push_back(args[i]);
  }
  *ret = VMTuple(fields);
});

TVM_REGISTER_API("relay._vm._Datatype")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    *ret = VMTensor(args[0]);
});


TVM_REGISTER_API("relay._vm._evaluate_vm")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    NodeRef to_compile = args[0];
    TVMContext ctx;
    int dev_type = args[1];
    ctx.device_type = static_cast<DLDeviceType>(dev_type);
    ctx.device_id = args[2];

    Module module;
    if (to_compile.as<FunctionNode>()) {
      Function to_compile = args[0];
      module = ModuleNode::FromExpr(to_compile);
    } else if (to_compile.as<ModuleNode>()) {
      module = args[0];
    } else {
      LOG(FATAL) << "expected function or module";
    }

    std::vector<VMObject> vm_args;
    for (auto i = 3; i < args.size(); i++) {
      VMObject obj = args[i];
      vm_args.push_back(obj);
    }
    auto result = EvaluateModule(module, {ctx}, vm_args);
    std::cout << "Returning results\n";
    *ret = VMToValue(std::get<1>(result), std::get<0>(result));
});


}  // namespace vm
}  // namespace relay
}  // namespace tvm
