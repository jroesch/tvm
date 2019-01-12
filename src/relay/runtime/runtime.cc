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
      this->packed_index = instr.packed_index;
      this->arity = instr.arity;
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

Instruction InvokePacked(size_t packed_index, size_t arity) {
  Instruction instr;
  instr.op = Opcode::InvokePacked;
  instr.packed_index = packed_index;
  instr.arity = arity;
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
      CHECK(func_node);

      std::cout << "Pushing Arguments on to stack" << std::endl;
      // First generate instructions to populate stack with arguments.
      std::cout << call_node->args << std::endl;
      for (auto arg : call_node->args) {
        this->VisitExpr(arg);
      }

      std::cout << "Allocating space for return value" << std::endl;
      // Allocate space for the return tensor.
      Type rtype = call_node->checked_type();
      std::cout << "Return type: " << rtype << std::endl;
      const TensorTypeNode* ttype = rtype.as<TensorTypeNode>();
      CHECK(ttype);

      std::vector<int64_t> shapes;
      std::cout << "generating shape" << std::endl;
      for (auto sh : ttype->shape) {
        shapes.push_back(Downcast<tvm::Integer>(sh)->value);
      }

      DataType dtype = ttype->dtype;
      TVMType dltype = Type2TVMType(dtype);
      Instruction alloc = AllocTensor(shapes, dltype);
      Emit(alloc);

      std::cout << "Emit invoke" << std::endl;
      // Next generate the invoke instruction.
      CHECK(func_node->IsPrimitive());
      auto target = Target::create("llvm");
      auto key = CCacheKeyNode::make(GetRef<Function>(func_node), target);
      auto cfunc = engine->Lower(key);
      // TODO: support lowered funcs for multiple targets
      CHECK(cfunc->funcs.size() == 1);
      auto op_index = this->lowered_funcs.size();
      this->lowered_funcs.push_back(cfunc->funcs[0]);
      // TODO(@jroesch): this doesn't support tuples right now.
      size_t arity = func_node->params.size() + 1;
      CHECK(arity < 10);
      Emit(InvokePacked(op_index, arity));
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
  std::cout << "final stack size: " << stack.size() << "bp: " << bp << std::endl;
  Run();
  std::cout << "final stack size: " << stack.size() << "bp: " << bp << std::endl;
  return stack.back();
}

void InvokePacked(const PackedFunc& func, size_t arg_count, std::vector<VMObject>& stack) {
  CHECK(arg_count <= stack.size());

  std::vector<TVMValue> values(arg_count);
  std::vector<int> codes(arg_count);
  runtime::TVMArgsSetter setter(values.data(), codes.data());

  std::cout << "InvokePacked: " << stack.size();

  auto stack_start = stack.size() - arg_count - 1;
  for (size_t i = 0; i < arg_count; i++) {
    std::cout << "Getting: " << stack_start + i << std::endl;
    NDArray data = ToNDArray(stack[stack_start + i]);
    setter(i, data);
  }

  TVMRetValue rv;
  func.CallPacked(TVMArgs(values.data(), codes.data(), arg_count), &rv);
  stack[stack.size() - arg_count - 2] = stack[stack.size() - 1];
  stack.resize(stack.size() - arg_count);
}

void VirtualMachine::Run() {
  CHECK(this->code);
  this->pc = 0;
  auto stack_start = frames.size();
  while (true) {
  main_loop:
    auto const& instr = this->code[this->pc];
    std::cout << "Executing: ";
    InstructionPrint(std::cout, instr);
    std::cout << std::endl;
    switch (instr.op) {
      case Opcode::InvokePacked: {
        const auto& func = packed_funcs[instr.packed_index];
        const auto& arity = instr.arity;
        std::cout << "before call" << std::endl;
        std::cout << this->stack.size() << std::endl;
        InvokePacked(func, arity, stack);
        std::cout << "after call" << std::endl;
        pc++;
        goto main_loop;
      }
      case Opcode::AllocTensor: {
        const auto& ti = instr.tensor_info;
        DLContext ctx;
        ctx.device_type = DLDeviceType::kDLCPU;
        ctx.device_id = 0;
        auto shape = std::vector<int64_t>(ti.ndim);
        shape.assign(ti.shape, ti.shape + ti.ndim);
        auto data = NDArray::Empty(shape, ti.dtype, ctx);
        stack.push_back(VMTensor(data));
        pc++;
        goto main_loop;
      }
      case Opcode::Push: {
        CHECK(bp + instr.stack_index < stack.size());
        stack.push_back(stack[bp + instr.stack_index]);
        pc++;
        goto main_loop;
      }
      case Opcode::Ret: {
        std::cout << "ret";
        // If we have hit the point from which we started
        // running, we should return to the caller breaking
        // the dispatch loop.
        if (PopFrame() == stack_start) {
          std::cout << "finish";
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

TVM_REGISTER_API("relay._runtime._testeval")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    Function to_compile = args[0];
    tvm::Array<Value> vargs = args[1];
    VirtualMachine vm = CompileFunc(to_compile);
    VMFunctionPrint(vm.functions[0]);
    std::cout << "Before convert" << std::endl;
    TensorValue tv = Downcast<TensorValue>(vargs[0]);
    VMObject vm_tensor = VMTensor(tv->data);
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
