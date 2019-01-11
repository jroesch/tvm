/*!
 *  Copyright (c) 2018 by Contributors
 * \file tvm/relay/runtime/runtime.h
 * \brief Abstract device memory management API
 */

#include <tvm/relay/expr_functor.h>
#include <tvm/relay/runtime/runtime.h>
#include <tvm/relay/interpreter.h>
#include "../backend/compile_engine.h"

#include <vector>
#include <iostream>

namespace tvm {
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
    case Opcode::InvokePacked:
      this->stack_index = instr.stack_index;
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

Instruction InvokePacked(size_t stack_index) {
  Instruction instr;
  instr.op = Opcode::InvokePacked;
  instr.stack_index = stack_index;
  return instr;
}

Instruction AllocTensor(std::vector<int64_t> shape, DLDataType dtype) {
  Instruction instr;
  instr.op = Opcode::AllocTensor;
  instr.tensor_info.shape = shape;
  instr.tensor_info.dtype = dtype;
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
      os << "invoke_packed";
      break;
    }
    case Opcode::AllocTensor: {
      os << "alloc_tensor";
      os << "(";
      for (auto sh : instr.tensor_info.shape) {
        os << sh << ", ";
      }
      os << ") ";
      os << TVMType2Type(instr.tensor_info.dtype);
      break;
    }
  }
}

void VMFunctionPrint(const VMFunction& vm_func) {
  for (auto instr : vm_func.instructions) {
    InstructionPrint(std::cout, instr);
    std::cout << ";" << std::endl;
  }
}

struct VMCompiler : ExprFunctor<void(const Expr& expr)> {
    std::vector<Instruction> instructions;
    std::unordered_map<Var, size_t, NodeHash, NodeEqual> var_map;
    size_t stack_index;
    bool seen_func;
    CompileEngine engine;
    std::vector<LoweredFunc> lowered_funcs;

    VMCompiler() :
      instructions(), var_map(), stack_index(0),
      seen_func(false), engine(CompileEngine::Global()) {}

    inline void Emit(const Instruction& instr) {
      instructions.push_back(instr);
    }

    void VisitExpr_(const VarNode* var_node) {
      auto var = GetRef<Var>(var_node);
      auto it = this->var_map.find(var);
      CHECK(it != this->var_map.end());
      this->instructions.push_back(Push(it->second));
    }

    void VisitExpr_(const CallNode* call_node) {
      auto func_node = call_node->op.as<FunctionNode>();
      // First generate instructions to populate stack with arguments.
      for (auto arg : call_node->args) {
        this->VisitExpr(arg);
      }

      // Allocate space for the return tensor.
      Type rtype = call_node->checked_type();
      const TensorTypeNode* ttype = rtype.as<TensorTypeNode>();
      CHECK(ttype);
      std::vector<int64_t> shapes;
      for (auto sh : ttype->shape) {
        shapes.push_back(Downcast<tvm::Integer>(sh)->value);
      }

      auto dltype = Type2TVMType(ttype->dtype);
      Emit(AllocTensor(shapes, dltype));
      // Next generate the invoke instruction.
      CHECK(func_node);
      CHECK(func_node->IsPrimitive());
      auto target = Target::create("llvm");
      auto key = CCacheKeyNode::make(GetRef<Function>(func_node), target);
      auto cfunc = engine->Lower(key);
      // TODO: support lowered funcs for multiple targets
      CHECK(cfunc->funcs.size() == 1);
      auto op_index = this->lowered_funcs.size();
      this->lowered_funcs.push_back(cfunc->funcs[0]);
      this->instructions.push_back(InvokePacked(op_index));
    }

    void VisitExpr_(const FunctionNode* func_node) {
      CHECK(!seen_func);
      this->seen_func = true;
      for (auto param : func_node->params) {
        var_map.insert({ param, this->stack_index++ });
      }

      this->VisitExpr(func_node->body);
    }
};

VirtualMachine CompileFunc(const Function& func) {
  size_t params = func->params.size();
  VMCompiler compiler;
  compiler.VisitExpr(func);
  compiler.instructions.push_back(Ret());
  auto main_func = VMFunction(params, compiler.instructions);
  runtime::Module mod;
  VirtualMachine vm;
  if (compiler.lowered_funcs.size() > 0) {
    Target target = Target::create("llvm");
    if (const auto* f = runtime::Registry::Get("relay.backend.build")) {
      mod = (*f)(tvm::Array<LoweredFunc>(compiler.lowered_funcs.begin(), compiler.lowered_funcs.end()), target);
    } else {
      LOG(FATAL) << "relay.backend.build is not registered";
    }
    CHECK(mod.operator->());
    for (auto lfunc : compiler.lowered_funcs) {
      vm.packed_funcs.push_back(mod.GetFunction(lfunc->name));
    }
  }
  vm.functions.push_back(main_func);

  return vm;
}

void VirtualMachine::PushFrame(size_t arg_count, size_t ret_pc, const VMFunction& vm_func) {
  auto frame = VMFrame(ret_pc, bp, func_index, arg_count, code);
  frames.push_back(frame);
  std::cout << "initial stack size" << stack.size() << std::endl;
}

size_t VirtualMachine::PopFrame() {
  CHECK(frames.size() != 0);
  const VMFrame& fr = frames.back();
  auto stack_size = stack.size();
  // Copy return value;
  stack[stack_size - fr.args - 1] = stack[stack_size - 1];
  // Resize value stack.
  stack.resize(stack_size - fr.args);
  // Reset frame.
  bp = fr.bp;
  pc = fr.pc;
  func_index = fr.func_index;
  code = fr.code;
  auto call_stack_size = frames.size();
  frames.pop_back();
  return call_stack_size;
}

VMObject VirtualMachine::Invoke(VMFunction func, std::vector<VMObject> args) {
  CHECK(args.size() == func.params);
  stack.push_back(VMObject());
  for (auto arg : args) {
    stack.push_back(arg);
  }
  PushFrame(func.params, this->pc + 1, func);
  code = func.instructions.data();
  pc = 0;
  bp = stack.size() - func.params;
  Run();
  std::cout << "final stack size: " << stack.size() << std::endl;
  return stack.back();
}

void VirtualMachine::Run() {
  CHECK(this->code);
  this->pc = 0;
  auto stack_start = frames.size();
  while (true) {
  main_loop:
    auto const& instr = this->code[this->pc];
    switch (instr.op) {
      case Opcode::InvokePacked:
        LOG(FATAL) << "YOOOOOO";
      case Opcode::AllocTensor:
        LOG(FATAL) << "ALLOC";
      case Opcode::Push:
        CHECK(bp + instr.stack_index < stack.size());
        stack.push_back(stack[bp + instr.stack_index]);
        pc++;
        goto main_loop;
      case Opcode::Ret:
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

TVM_REGISTER_API("relay._runtime._testeval")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    Function to_compile = args[0];
    tvm::Array<Value> vargs = args[1];
    VirtualMachine vm = CompileFunc(to_compile);
    VMFunctionPrint(vm.functions[0]);
    std::cout << "Before convert" << std::endl;
    TensorValue tv = Downcast<TensorValue>(vargs[0]);
    auto vm_tensor = VMTensor(tv->data);
    std::cout << "before invoke" << std::endl;
    VMObject result = vm.Invoke(vm.functions[0], { vm_tensor });
    std::cout << "testing eval" << std::endl;
    // directly returning ndarray causes segfault
    NDArray nd = ToNDArray(result);
    std::cout << "Getting ND finished";
    *ret = TensorValueNode::make(nd);
});


}  // namespace vm
}  // namespace relay
}  // namespace tvm
