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
#include <chrono>

using namespace tvm::runtime;

namespace tvm {

// Packed Function extensions.
TVMRetValue& runtime::TVMRetValue::operator=(relay::vm::Object other) {
  this->SwitchToClass(kObject, other);
  return *this;
}

runtime::TVMArgValue::operator relay::vm::Object() const {
  if (type_code_ == kNull) return relay::vm::Object(nullptr);
  TVM_CHECK_TYPE_CODE(type_code_, kObject);
  return *ptr<relay::vm::Object>();
}

namespace relay {
namespace vm {


Instruction::Instruction() {}

Instruction::Instruction(const Instruction& instr) {
  this->op = instr.op;
  this->dst = instr.dst;

  switch (instr.op) {
    case Opcode::Move:
      this->from = instr.from;
      return;
    case Opcode::Select:
      this->select_cond = instr.select_cond;
      this->select_op1 = instr.select_op1;
      this->select_op2 = instr.select_op2;
      return;
    case Opcode::Ret:
      this->result = instr.result;
      return;
    case Opcode::AllocTensor:
      this->tensor_info = instr.tensor_info;
      return;
    case Opcode::AllocDatatype:
      this->constructor_tag = instr.constructor_tag;
      this->num_fields = instr.num_fields;
      this->datatype_fields = instr.datatype_fields;
      return;
    case Opcode::AllocClosure:
      this->clo_index = instr.clo_index;
      this->num_freevar = instr.num_freevar;
      this->free_vars = instr.free_vars;
      return;
    case Opcode::InvokePacked:
      this->packed_index = instr.packed_index;
      this->arity = instr.arity;
      this->output_size = instr.output_size;
      this->packed_args = instr.packed_args;
      return;
    case Opcode::InvokeClosure:
      this->closure = instr.closure;
      this->closure_args_num = instr.closure_args_num;
      this->closure_args = instr.closure_args;
      return;
    case Opcode::If:
      this->if_cond = instr.if_cond;
      this->true_offset = instr.true_offset;
      this->false_offset = instr.false_offset;
      return;
    case Opcode::Invoke:
      this->func_index = instr.func_index;
      this->num_args = instr.num_args;
      this->invoke_args_registers = instr.invoke_args_registers;
      return;
    case Opcode::LoadConst:
      this->const_index = instr.const_index;
      return;
    case Opcode::GetField:
      this->object = instr.object;
      this->field_index = instr.field_index;
      return;
    case Opcode::Goto:
      this->pc_offset = instr.pc_offset;
      return;
  }
}

// TODO(@jroesch): this leaks memory fix me
Instruction::~Instruction() {}

Instruction Ret(VirtualRegisterNum result) {
  Instruction instr;
  instr.op = Opcode::Ret;
  instr.result = result;
  return instr;
}

Instruction InvokePacked(size_t packed_index, size_t arity, size_t output_size, const std::vector<VirtualRegisterNum>& args) {
  Instruction instr;
  instr.op = Opcode::InvokePacked;
  instr.packed_index = packed_index;
  instr.arity = arity;
  instr.output_size = output_size;
  instr.packed_args = new VirtualRegisterNum[arity];
  for (int i = 0; i < arity; ++i) {
    instr.packed_args[i] = args[i];
  }
  return instr;
}

Instruction AllocTensor(const std::vector<int64_t>& shape, DLDataType dtype, size_t dst) {
  Instruction instr;
  instr.op = Opcode::AllocTensor;
  instr.dst = dst;
  instr.tensor_info.shape = new int64_t[shape.size()];
  instr.tensor_info.ndim = shape.size();
  std::memcpy(
      reinterpret_cast<void*>(instr.tensor_info.shape),
      reinterpret_cast<const void*>(shape.data()),
      shape.size() * sizeof(int64_t));
  instr.tensor_info.dtype = dtype;
  return instr;
}

Instruction AllocDatatype(size_t tag, size_t num_fields, const std::vector<VirtualRegisterNum>& datatype_fields, size_t dst) {
  Instruction instr;
  instr.op = Opcode::AllocDatatype;
  instr.dst = dst;
  instr.constructor_tag = tag;
  instr.num_fields = num_fields;
  instr.datatype_fields = new VirtualRegisterNum[num_fields];
  for (auto i = 0; i < num_fields; ++i) {
    instr.datatype_fields[i] = datatype_fields[i];
  }
  return instr;
}

Instruction AllocClosure(size_t func_index, size_t free_vars, const std::vector<VirtualRegisterNum>& free_var_register, size_t dst) {
  Instruction instr;
  instr.op = Opcode::AllocClosure;
  instr.dst = dst;
  instr.clo_index = func_index;
  instr.num_freevar = free_vars;
  instr.free_vars = new VirtualRegisterNum[instr.num_freevar];
  for (size_t i = 0; i < instr.num_freevar; ++i) {
    instr.free_vars[i] = free_var_register[i];
  }
  return instr;
}

Instruction GetField(VirtualRegisterNum object, size_t field_index, VirtualRegisterNum dst) {
  Instruction instr;
  instr.op = Opcode::GetField;
  instr.dst = dst;
  instr.object = object;
  instr.field_index = field_index;
  return instr;
}

Instruction If(VirtualRegisterNum cond, size_t true_branch, size_t false_branch) {
  Instruction instr;
  instr.op = Opcode::If;
  instr.if_cond = cond;
  instr.true_offset = true_branch;
  instr.false_offset = false_branch;
  return instr;
}

Instruction Select(VirtualRegisterNum cond, VirtualRegisterNum op1, VirtualRegisterNum op2, VirtualRegisterNum dst) {
  Instruction instr;
  instr.op = Opcode::Select;
  instr.dst = dst;
  instr.select_cond = cond;
  instr.select_op1 = op1;
  instr.select_op2 = op2;
  return instr;
}

Instruction Goto(size_t pc_offset) {
  Instruction instr;
  instr.op = Opcode::Goto;
  instr.pc_offset = pc_offset;
  return instr;
}

Instruction Invoke(size_t func_index, const std::vector<VirtualRegisterNum>& args_registers, VirtualRegisterNum dst) {
  Instruction instr;
  instr.op = Opcode::Invoke;
  instr.dst = dst;
  instr.func_index = func_index;
  instr.num_args = args_registers.size();
  instr.invoke_args_registers = new VirtualRegisterNum[instr.num_args];
  for (auto i = 0; i < instr.num_args; ++i) {
    instr.invoke_args_registers[i] = args_registers[i];
  }
  return instr;
}

Instruction InvokeClosure(VirtualRegisterNum closure, const std::vector<VirtualRegisterNum>& args, VirtualRegisterNum dst) {
  Instruction instr;
  instr.op = Opcode::InvokeClosure;
  instr.dst = dst;
  instr.closure = closure;
  instr.closure_args_num = args.size();
  instr.closure_args = new VirtualRegisterNum[args.size()];
  for (auto i = 0; i < args.size(); ++i) {
    instr.closure_args[i] = args[i];
  }
  return instr;
}

Instruction LoadConst(size_t const_index, VirtualRegisterNum dst) {
  Instruction instr;
  instr.op = Opcode::LoadConst;
  instr.dst = dst;
  instr.const_index = const_index;
  return instr;
}

Instruction Move(VirtualRegisterNum src, VirtualRegisterNum dst) {
  Instruction instr;
  instr.op = Opcode::Move;
  instr.dst = dst;
  instr.from = src;
  return instr;
}

void InstructionPrint(std::ostream& os, const Instruction& instr) {
  switch (instr.op) {
    case Opcode::Move: {
      os << "move " 
         << instr.from << " " 
         << instr.dst;
      break;
    }
    case Opcode::Ret: {
      os << "ret " 
         << instr.result;
      break;
    }
    case Opcode::InvokePacked: {
      os << "invoke_packed ";
      os << instr.packed_index;
      os << " " << instr.arity;
      os << "(";
      for (size_t i = 0; i < instr.arity; ++i) {
        os << instr.packed_args[i] << ",";
      }
      os << ")";
      os << " " << instr.output_size;  
      break;
    }
    case Opcode::AllocTensor: {
      os << "alloc_tensor ";
      os << instr.dst << " ";
      os << "(";

      for (size_t i = 0; i < instr.tensor_info.ndim; i++) {
        os << instr.tensor_info.shape[i] << ", ";
      }
      os << ") ";
      os << TVMType2Type(instr.tensor_info.dtype);
      break;
    }
    case Opcode::AllocDatatype: {
      os << "alloc_data ";
      os << instr.dst << " ";
      os << instr.constructor_tag << " ";
      os << instr.num_fields;
      break;
    }
    case Opcode::AllocClosure: {
      os << "alloc_closure ";
      os << instr.dst << " ";
      os << instr.clo_index << " ";
      os << instr.num_freevar << "(";
      for (size_t i = 0; i < instr.num_freevar; ++i) {
        os << instr.free_vars[i] << ",";
      }
      os << ")";
      break;
    }
    case Opcode::If: {
      os << "if "
         << "$" << instr.if_cond << " "
         << instr.true_offset << " "
         << instr.false_offset;
      break;
    }
    case Opcode::Invoke: {
      os << "invoke "
         << "$" << instr.dst << " "
         << instr.func_index << " "
         << instr.num_args << "(";
         for (size_t i = 0; i < instr.num_args; ++i) {
           os << instr.invoke_args_registers[i] << ",";
         }
         os << ")";
      break;
    }
    case Opcode::InvokeClosure: {
      os << "invoke_closure "
         << "$" << instr.dst << " "
         << instr.closure << " "
         << instr.closure_args_num << "()";
      break;
    }
    case Opcode::LoadConst: {
      os << "load_const "
         << "$" << instr.dst << " "
         << instr.const_index;
      break;
    }
    case Opcode::GetField: {
      os << "get_field "
         << instr.dst << " "
         << instr.object << " "
         << instr.field_index;
      break;
    }
    case Opcode::Goto: {
      os << "goto "
         << instr.pc_offset;
      break;
    }
    case Opcode::Select: {
      os << "phi "
         << instr.dst << " "
         << instr.select_cond << " "
         << instr.select_op1 << " "
         << instr.select_op2;
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

void VirtualMachine::PushFrame(size_t arg_count, size_t ret_pc, const VMFunction& vm_func) {
  auto frame = VMFrame(ret_pc, func_index, arg_count, code, vm_func.register_file_size);
  frames.push_back(frame);
}

size_t VirtualMachine::PopFrame() {
  CHECK(frames.size() != 0);
  const VMFrame& fr = frames.back();
  func_index = fr.func_index;
  code = fr.code;
  pc = fr.pc;
  auto call_stack_size = frames.size();
  frames.pop_back();
  return call_stack_size;
}

void VirtualMachine::InvokeGlobal(const VMFunction& func, const std::vector<Object>& args) {
  RELAY_LOG(INFO) << "===================\nInvoking global " << func.name << " " << args.size()
                  << std::endl;

  PushFrame(func.params, this->pc + 1, func);
  for (size_t i = 0; i < args.size(); ++i) {
    WriteRegister(i+1, args[i]);
  }
  RELAY_LOG(INFO) << "func.params= " << func.params << std::endl;

  code = func.instructions.data();
  pc = 0;
}

Object VirtualMachine::Invoke(const VMFunction& func, const std::vector<Object>& args) {
  RELAY_LOG(INFO) << "Executing function " << func.name << std::endl;

  InvokeGlobal(func, args);
  Run();
  auto alloc = MemoryManager::Global()->GetAllocator(ctxs[0]);
  RELAY_LOG(INFO) << "Memory used: " << alloc->UsedMemory() << " B\n";
  return return_register;
}

Object VirtualMachine::Invoke(const GlobalVar& global,
                                const std::vector<Object>& args) {
  auto func_index = this->global_map[global];
  RELAY_LOG(INFO) << "Invoke Global " << global << " at index " << func_index
                  << std::endl;
  return Invoke(this->functions[func_index], args);
}

void InvokePacked(const PackedFunc& func, size_t arg_count, size_t output_size,
                  std::vector<Object>& args) {
  std::vector<TVMValue> values(arg_count);
  std::vector<int> codes(arg_count);
  runtime::TVMArgsSetter setter(values.data(), codes.data());

  for (size_t i = 0; i < arg_count; i++) {
    NDArray data = ToNDArray(args[i]);
    setter(i, data);
  }

  TVMRetValue rv;
  func.CallPacked(TVMArgs(values.data(), codes.data(), arg_count), &rv);
}

void VirtualMachine::Init(const std::vector<TVMContext>& ctxs) {
  this->ctxs = ctxs;
}

inline void VirtualMachine::WriteRegister(size_t r, Object val) {
  frames.back().register_file[r] = val;
}

inline Object VirtualMachine::ReadRegister(size_t r) {
  return frames.back().register_file[r];
}

void VirtualMachine::Run() {
  CHECK(this->code);
  this->pc = 0;
  auto frame_start = frames.size();
  while (true) {
  main_loop:
    auto const& instr = this->code[this->pc];
    RELAY_LOG(INFO) << "\nExecuting(" << pc << "): " ;
#if USE_RELAY_LOG
    InstructionPrint(std::cout, instr);
#endif  // USE_RELAY_LOG

    switch (instr.op) {
      case Opcode::Move: {
        Object from_obj;
        if (instr.from == 0) {
          from_obj = return_register;
        } else {
          from_obj = ReadRegister(instr.from);
        }
        WriteRegister(instr.dst, from_obj);
        pc++;
        goto main_loop;
      }
      case Opcode::LoadConst: {
        WriteRegister(instr.dst, this->constants[instr.const_index]);
        pc++;
        goto main_loop;
      }
      case Opcode::Invoke: {
        std::vector<Object> args;        
        for (size_t i = 0; i < instr.num_args; ++i) {
          args.push_back(ReadRegister(instr.invoke_args_registers[i]));
        }
        InvokeGlobal(this->functions[instr.func_index], args);
        goto main_loop;
      }
      case Opcode::InvokePacked: {
        const auto& func = packed_funcs[instr.packed_index];
        const auto& arity = instr.arity;
        std::vector<Object> args;
        for (size_t i = 0; i < arity; ++i) {
          args.push_back(ReadRegister(instr.packed_args[i]));
        }
        InvokePacked(func, arity, instr.output_size, args);
        for (size_t i = 0; i < instr.output_size; ++i) {
          WriteRegister(instr.packed_args[instr.arity - instr.output_size + i], args[instr.arity - instr.output_size + i]);
        }
        pc++;
        goto main_loop;
      }
      case Opcode::InvokeClosure: {
        auto object = ReadRegister(instr.closure);
        CHECK(object->tag == ObjectTag::kClosure);
        const std::shared_ptr<ClosureCell>& closure = std::dynamic_pointer_cast<ClosureCell>(object.ptr);
        std::vector<Object> args;
        for (size_t i = 0; i < instr.closure_args_num; ++i) {
          args.push_back(ReadRegister(instr.closure_args[i]));
        }
        for (auto free_var : closure->free_vars) {
          args.push_back(free_var);
        }
        InvokeGlobal(this->functions[closure->func_index], args);
        goto main_loop;
      }
      case Opcode::GetField: {
        auto object = ReadRegister(instr.object);
        CHECK(object->tag == ObjectTag::kDatatype) << "Object is not data type object, register " << instr.object << ", Object tag " << (int)object->tag;
        const std::shared_ptr<DatatypeCell>& tuple = std::dynamic_pointer_cast<DatatypeCell>(object.ptr);
        auto field = tuple->fields[instr.field_index];
        WriteRegister(instr.dst, field);
        pc++;
        goto main_loop;
      }
      case Opcode::Goto: {
        pc += instr.pc_offset;
        goto main_loop;
      }
      case Opcode::If: {
        // How do we do this efficiently?
        DLContext cpu_ctx;
        cpu_ctx.device_type = kDLCPU;
        cpu_ctx.device_id = 0;

        const auto& cond = ReadRegister(instr.if_cond);
        NDArray cpu_array = ToNDArray(cond).CopyTo(cpu_ctx);
        CHECK_EQ(TVMType2Type(cpu_array->dtype), Bool());
        bool branch = reinterpret_cast<uint8_t*>(cpu_array->data)[0];

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
        auto obj = TensorObj(data);
        WriteRegister(instr.dst, obj);
        pc++;
        goto main_loop;
      }
      case Opcode::AllocDatatype: {
        std::vector<Object> fields;
        for (size_t i = 0; i < instr.num_fields; ++i) {
          fields.push_back(ReadRegister(instr.datatype_fields[i]));
        }
        Object obj = DatatypeObj(instr.constructor_tag, fields);
        WriteRegister(instr.dst, obj);
        pc++;
        goto main_loop;
      }
      case Opcode::AllocClosure: {
        std::vector<Object> free_vars;
        for (size_t i = 0; i < instr.num_freevar; i++) {
          free_vars.push_back(ReadRegister(instr.free_vars[i]));
        }
        WriteRegister(instr.dst, ClosureObj(instr.func_index, free_vars));
        pc++;
        goto main_loop;
      }
      case Opcode::Select: {
        DLContext cpu_ctx;
        cpu_ctx.device_type = kDLCPU;
        cpu_ctx.device_id = 0;

        auto cond = ReadRegister(instr.select_cond);
        NDArray cpu_array = ToNDArray(cond).CopyTo(cpu_ctx);
        CHECK_EQ(TVMType2Type(cpu_array->dtype), Bool());
        bool branch = reinterpret_cast<uint8_t*>(cpu_array->data)[0];

        if (branch) {
          auto op1 = ReadRegister(instr.select_op1);
          WriteRegister(instr.dst, op1);
        } else {
          auto op2 = ReadRegister(instr.select_op2);
          WriteRegister(instr.dst, op2);
        }
        pc++;
        goto main_loop;
      }
      case Opcode::Ret: {
        // If we have hit the point from which we started
        // running, we should return to the caller breaking
        // the dispatch loop.
        return_register = ReadRegister(instr.result);

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
void ConvertArgsToVM(tvm::Array<Value> args, std::vector<Object>& out) {
  for (auto arg : args) {
    if (auto tensor = arg.as<TensorValueNode>()) {
      out.push_back(TensorObj(tensor->data));
    } else if (auto tuple = arg.as<TupleValueNode>()) {
      std::vector<Object> fields;
      for (auto field : tuple->fields) {
        ConvertArgsToVM({field}, fields);
      }
      out.push_back(DatatypeObj(0, fields));
    } else {
      LOG(FATAL) << "unknown case: " << arg;
    }
  }
}

/*! \brief Convert from an array of relay.Value into VM compatible objects.
 */
Object ValueToVM(Value value) {
  std::vector<Object> out;
  ConvertArgsToVM({value}, out);
  CHECK_LT(out.size(), 2);
  return out[0];
}

using TagNameMap = std::unordered_map<size_t, tvm::relay::Constructor>;

Value VMToValue(TagNameMap& tag_index_map, Object obj) {
  switch (obj->tag) {
    case ObjectTag::kTensor: {
      return TensorValueNode::make(ToNDArray(obj));
    }
    case ObjectTag::kDatatype: {
      auto data_type = std::dynamic_pointer_cast<DatatypeCell>(obj.ptr);

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

std::tuple<Object, TagNameMap>
EvaluateModule(const Module& module, const std::vector<TVMContext> ctxs,
               const std::vector<Object>& vm_args) {
  VirtualMachine vm = VirtualMachine::FromModule(module, ctxs);
  //TODO(zhiics) This measurement is for temporary usage. Remove it later. We
  //need to introduce a better profiling method.
#if ENABLE_PROFILING
  RELAY_LOG(INFO) << "Entry function is " << module->entry_func << std::endl;
  auto start = std::chrono::high_resolution_clock::now();
#endif  // ENABLE_PROFILING
  std::tuple<Object, TagNameMap> res =
      std::make_tuple(vm.Invoke(module->entry_func, vm_args), vm.tag_index_map);
#if ENABLE_PROFILING
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  LOG(INFO) << "Inference time: " << duration << "ms\n";
#endif  // ENABLE_PROFILING
  return res;
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
    *ret = TensorObj(args[0]);
});

TVM_REGISTER_API("relay._vm._Tuple")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  std::vector<Object> fields;
  for (auto i = 0; i < args.size(); i++) {
    fields.push_back(args[i]);
  }
  *ret = TupleObj(fields);
});

template<typename T>
std::string ToString(const T& t) {
  std::stringstream s;
  s << t;
  return s.str();
}

TVM_REGISTER_API("relay._vm._ObjectTag")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  Object obj = args[0];
  *ret = ToString(obj->tag);
});

// TVM_REGISTER_API("relay._vm._Datatype")
// .set_body([](TVMArgs args, TVMRetValue* ret) {
//     *ret = DatatypeObj(args[0]);
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

    std::vector<Object> vm_args;
    for (auto i = 3; i < args.size(); i++) {
      Object obj = args[i];
      vm_args.push_back(obj);
    }

    auto result = EvaluateModule(module, {ctx}, vm_args);
    RELAY_LOG(INFO) << "Returning results\n";
    *ret = VMToValue(std::get<1>(result), std::get<0>(result));
});


}  // namespace vm
}  // namespace relay
}  // namespace tvm
