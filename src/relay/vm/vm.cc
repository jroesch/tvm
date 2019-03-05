/*!
 *  Copyright (c) 2018 by Contributors
 * \file tvm/relay/runtime/runtime.h
 * \brief Abstract device memory management API
 */

#include <tvm/runtime/memory_manager.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/vm/vm.h>
#include <tvm/relay/interpreter.h>
#include <tvm/relay/logging.h>
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
    case Opcode::AllocClosure:
      this->clo_index = instr.clo_index;
      this->num_freevar = instr.num_freevar;
      return;
    case Opcode::InvokePacked:
      this->packed_index = instr.packed_index;
      this->arity = instr.arity;
      this->output_size = instr.output_size;
      return;
    case Opcode::InvokeClosure:
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
    case Opcode::Move:
      this->source = instr.source;
      this->dest = instr.dest;
      return;
    case Opcode::Pop:
      this->pop_count = instr.pop_count;
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

Instruction Pop(size_t pop_count) {
  Instruction instr;
  instr.op = Opcode::Pop;
  instr.pop_count = pop_count;
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

Instruction AllocTensor(const std::vector<int64_t>& shape, DLDataType dtype) {
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

Instruction AllocClosure(size_t func_index, size_t free_vars) {
  Instruction instr;
  instr.op = Opcode::AllocClosure;
  instr.clo_index = func_index;
  instr.num_freevar = free_vars;
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

Instruction InvokeClosure() {
  Instruction instr;
  instr.op = Opcode::InvokeClosure;
  return instr;
}

Instruction LoadConst(size_t const_index) {
  Instruction instr;
  instr.op = Opcode::LoadConst;
  instr.const_index = const_index;
  return instr;
}

Instruction Move(size_t source, size_t dest) {
  Instruction instr;
  instr.op = Opcode::Move;
  instr.source = source;
  instr.dest = dest;
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
      os << "alloc_data";
      os << " ";
      os << instr.constructor_tag << " ";
      os << instr.num_fields;
      break;
    }
    case Opcode::AllocClosure: {
      os << "alloc_closure";
      os << " ";
      os << instr.clo_index << " ";
      os << instr.num_freevar;
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
    case Opcode::InvokeClosure: {
      os << "invoke_closure";
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
    case Opcode::Move: {
      os << "move "
         << instr.source << " "
         << instr.dest;
      break;
    }
    case Opcode::Pop: {
      os << "pop "
         << instr.pop_count;
      break;
    }
    default:
      LOG(FATAL) << "should never hit this case" << static_cast<int>(instr.op);
      break;
  }
}

std::ostream& operator<<(std::ostream& os, const Instruction& instr) {
  InstructionPrint(os, instr);
  return os;
}

void VMFunctionPrint(std::ostream& os, const VMFunction& vm_func) {
  os << vm_func.name << ": " << std::endl;
  for (size_t i = 0; i < vm_func.instructions.size(); ++i) {
    os << i << ": ";
    InstructionPrint(os, vm_func.instructions[i]);
    os << ";" << std::endl;
  }
}

std::ostream& operator<<(std::ostream& os, const VMFunction& vm_func) {
  VMFunctionPrint(os, vm_func);
  return os;
}

void VirtualMachine::PushFrame(size_t arg_count, size_t ret_pc, size_t sp, const VMFunction& vm_func) {
  CHECK(sp <= stack.size());
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
  // Copy return value to the position past last function's frame

  VMObject return_value = stack.back();
  // stack[fr.sp] = stack[stack_size - 1];
  // Resize value stack.

  stack.resize(fr.sp);
  DumpStack();
  stack.resize(stack.size() - fr.args);
  DumpStack();
  stack.push_back(return_value);
  DumpStack();

  CHECK(stack.size() < stack_size + 1)
    << "stack_size before modifying stack = " << stack_size + 1
    << " stack size after modifying the stack = " << stack.size();

  // Reset frame.
  RELAY_LOG(INFO) << "Num Args: " << fr.args;
  RELAY_LOG(INFO) << "Reset frame " << bp << " -> " << fr.bp << "\n";
  RELAY_LOG(INFO) << "Stack pointer: " << fr.sp << std::endl;
  RELAY_LOG(INFO) << "Reset stack " << stack_size << " -> " << stack.size() << "\n";
  bp = fr.bp;
  pc = fr.pc;
  func_index = fr.func_index;
  code = fr.code;
  auto call_stack_size = frames.size();
  frames.pop_back();
  return call_stack_size;
}

void VirtualMachine::InvokeGlobal(const VMFunction& func) {
  RELAY_LOG(INFO) << "===================\nInvoking global " << func.name
                  << std::endl;
  PushFrame(func.params, this->pc + 1, stack.size(), func);
  RELAY_LOG(INFO) << "func.params= " << func.params
                  << ", stack.size()= " << stack.size() << std::endl;

  code = func.instructions.data();
  pc = 0;
  bp = stack.size() - func.params ;
}

VMObject VirtualMachine::Invoke(const VMFunction& func, const std::vector<VMObject>& args) {
  RELAY_LOG(INFO) << "Executing function " << func.name << " bp " << bp
                  << std::endl;
  for (auto arg : args) {
    stack.push_back(arg);
  }

  InvokeGlobal(func);
  Run();
  auto alloc = MemoryManager::Global()->GetAllocator(ctxs[0]);
  RELAY_LOG(INFO) << "Memory used: " << alloc->UsedMemory() << " B\n";
  RELAY_LOG(INFO) << "final stack size: " << stack.size() << "bp: " << bp
                  << std::endl;
  return stack.back();
}

VMObject VirtualMachine::Invoke(const GlobalVar& global, const std::vector<VMObject>& args) {
  auto func_index = this->global_map[global];
  RELAY_LOG(INFO) << "Invoke Global " << global << " at index " << func_index
                  << std::endl;
  return Invoke(this->functions[func_index], args);
}

void InvokePacked(const PackedFunc& func, size_t arg_count, size_t output_size, std::vector<VMObject>& stack) {
  auto stack_end = stack.size() - 1;
  RELAY_LOG(INFO) << "arg_count: " << arg_count;
  CHECK(arg_count <= stack.size());

  std::vector<TVMValue> values(arg_count);
  std::vector<int> codes(arg_count);
  runtime::TVMArgsSetter setter(values.data(), codes.data());

  auto argument_start = stack.size() - arg_count;
  RELAY_LOG(INFO) << "ArgumentStart=" << argument_start << std::endl;
  for (size_t i = 0; i < arg_count; i++) {
    NDArray data = ToNDArray(stack[argument_start + i]);
    setter(i, data);
  }

  TVMRetValue rv;
  func.CallPacked(TVMArgs(values.data(), codes.data(), arg_count), &rv);

  // // Fix the object at return value position
  // if (output_size == 1) {
  //   stack[stack.size() - 1] = stack[stack.size() - 2];
  // } else {
  //   auto adt = std::dynamic_pointer_cast<VMDatatypeCell>(stack.back().ptr);
  //   for (size_t i = 0; i < output_size; ++i) {
  //     adt->fields[i] = stack[stack.size() - output_size - 1 + i];
  //   }
  // }

  // We can do this more efficiently by reverse laying out the arguments
  // and just shrinking the stack.
  stack[stack.size() - arg_count] = stack[stack_end];
  RELAY_LOG(INFO) << "ShrinkBy=" << arg_count - output_size << std::endl;
  stack.resize(stack.size() - (arg_count - output_size));
}

void VirtualMachine::Init(const std::vector<TVMContext>& ctxs) {
  this->ctxs = ctxs;
}

template <typename T>
typename std::enable_if<T::value, void>::type
VirtualMachine::DumpRegister() {
  RELAY_LOG(INFO) << std::endl << "-- Registers: --\n";
  RELAY_LOG(INFO) << "Bp: " << bp << std::endl;
  RELAY_LOG(INFO) << "Stack Size: " << stack.size() << std::endl;
  RELAY_LOG(INFO) << "Frame Size: " << frames.size() << std::endl;
  RELAY_LOG(INFO) << "----\n" ;
}

template <typename T>
typename std::enable_if<T::value, void>::type VirtualMachine::DumpStack() {
  RELAY_LOG(INFO) << "DumpStack---\n";
  for (size_t i = bp; i < stack.size(); ++i) {
    RELAY_LOG(INFO) << i << " " << VMObjectTagString(stack[i]->tag) << " ";
    switch (stack[i]->tag) {
      case VMObjectTag::kTensor: {
        VMTensorCell* tensor = (VMTensorCell*)stack[i].operator->();
        RELAY_LOG(INFO) << "dimensions=" << tensor->data->ndim;
        if (tensor->data->ndim == 0) {
          RELAY_LOG(INFO) << " " << TensorValueNode::make(tensor->data);
        }
        RELAY_LOG(INFO) << " \n";
        break;
      }
      case VMObjectTag::kDatatype: {
        VMDatatypeCell* datatype = (VMDatatypeCell*)stack[i].operator->();
        std::cout << "fields: " << datatype->fields.size();
        std::cout << "\n";
        break;
      }
      default: {
        RELAY_LOG(INFO) << "\n";
      }
    }
  }
  RELAY_LOG(INFO) << "DumpStack end---\n";
}

void VirtualMachine::Run() {
  CHECK(this->code);
  this->debug = true;
  this->pc = 0;
  auto frame_start = frames.size();
  while (true) {
  main_loop:
    auto const& instr = this->code[this->pc];
    RELAY_LOG(INFO) << "Executing(" << pc << "): " ;
    InstructionPrint(std::cout, instr);
    RELAY_LOG(INFO) << "\n";
    DumpRegister();

    switch (instr.op) {
      case Opcode::LoadConst: {
        stack.push_back(this->constants[instr.const_index]);
        pc++;
        goto main_loop;
      }
      case Opcode::Invoke: {
        InvokeGlobal(this->functions[instr.func_index]);
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
        CHECK(start_stack - (arity - instr.output_size) == stack.size())
          << "start_stack: " << start_stack
          << " end_stack: " << stack.size();
        // std::cout << "after call" << std::endl;
        pc++;
        goto main_loop;
      }
      case Opcode::InvokeClosure: {
        auto object = stack.back();
        stack.pop_back();
        CHECK(object->tag == VMObjectTag::kClosure);
        const std::shared_ptr<VMClosureCell>& closure = std::dynamic_pointer_cast<VMClosureCell>(object.ptr);
        for (auto free_var : closure->free_vars) {
          stack.push_back(free_var);
        }
        InvokeGlobal(this->functions[closure->func_index]);
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
      case Opcode::AllocClosure: {
        std::vector<VMObject> free_vars;
        auto field_start = stack.size() - instr.num_freevar;
        // Optimize this code.
        for (size_t i = 0; i < instr.num_freevar; i++) {
          free_vars.push_back(stack[field_start + i]);
        }
        for (size_t i = 0; i < instr.num_freevar; i++) {
          stack.pop_back();
        }
        stack.push_back(VMClosure(instr.func_index, free_vars));
        DumpStack();
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
      case Opcode::Pop: {
        CHECK(bp + instr.pop_count < stack.size())
          << "bp=" << bp
          << " pop_count=" << instr.pop_count;
        auto new_size = stack.size() - instr.pop_count;
        stack.resize(new_size);
        DumpStack();
        pc++;
        goto main_loop;
      }
      case Opcode::Move: {
        CHECK(instr.source < stack.size())
          << "source=" << instr.source
          << " stack_size=" << stack.size();
        CHECK(instr.dest < stack.size())
          << "dest=" << instr.dest
          << " stack_size=" << stack.size();
        stack[bp + instr.dest] = stack[bp + instr.source];
        DumpStack();
        pc++;
        goto main_loop;
      }
      case Opcode::Ret: {
        // If we have hit the point from which we started
        // running, we should return to the caller breaking
        // the dispatch loop.
        DumpStack();
        if (PopFrame() == frame_start) {
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

using TagNameMap = std::unordered_map<size_t, tvm::relay::Constructor>;

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

      return ConstructorValueNode::make(tag_index_map[data_type->tag], fields);
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
  RELAY_LOG(INFO) << "Entry function is " << module->entry_func << std::endl;
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
  for (auto i = 0; i < args.size(); i++) {
    fields.push_back(args[i]);
  }
  *ret = VMTuple(fields);
});

TVM_REGISTER_API("relay._vm._VMObjectTag")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  VMObject obj = args[0];
  *ret = VMObjectTagString(obj->tag);
});

// TVM_REGISTER_API("relay._vm._Datatype")
// .set_body([](TVMArgs args, TVMRetValue* ret) {
//     *ret = VMDatatype(args[0]);
// });

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
    RELAY_LOG(INFO) << "Returning results\n";
    *ret = VMToValue(std::get<1>(result), std::get<0>(result));
});


}  // namespace vm
}  // namespace relay
}  // namespace tvm
