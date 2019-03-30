/*!
 *  Copyright (c) 2018 by Contributors
 * \file tvm/relay/runtime/runtime.h
 * \brief Abstract device memory management API
 */
#ifndef TVM_RELAY_RUNTIME_H_
#define TVM_RELAY_RUNTIME_H_

#include <vector>
#include <memory>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/logging.h>
#include <tvm/runtime/memory_manager.h>
#include <tvm/runtime/object.h>
#include <tvm/relay/vm/register_allocation.h>

namespace tvm {
namespace relay {
namespace vm {

using namespace tvm::runtime;

enum struct Opcode {
  Move,
  Ret,
  Invoke,
  InvokeClosure,
  InvokePacked,
  AllocTensor,
  AllocDatatype,
  AllocClosure,
  GetField,
  If,
  Phi,
  LoadConst,
  Goto
};

struct Instruction {
  struct TensorInfo {
      int64_t* shape;
      size_t ndim;
      DLDataType dtype;
  };

  Opcode op;

  // Destination register that the opcode writes to
  VirtualRegisterNum dst;

  union {
    size_t stack_index;
    TensorInfo tensor_info;

    // For InvokeClosure
    struct {
      VirtualRegisterNum closure;
      size_t closure_args_num;
      VirtualRegisterNum* closure_args;
    };
    // For Ret
    struct {
      VirtualRegisterNum result;
    };
    // For Move
    struct {
      VirtualRegisterNum from;
    };
    struct {
      size_t packed_index;
      size_t arity;
      size_t output_size;
      VirtualRegisterNum* packed_args;
    };
    // For Phi node
    struct {
      VirtualRegisterNum phi_cond;
      VirtualRegisterNum phi_op1;
      VirtualRegisterNum phi_op2;
    };
    // For If node
    struct {
      VirtualRegisterNum if_cond;
      size_t true_offset;
      size_t false_offset;
    };
    // For Invoke
    struct {
      size_t func_index;
      size_t num_args;
      VirtualRegisterNum* invoke_args_registers;
    };
    struct {
      size_t const_index;
    };
    struct {
      size_t pc_offset;
    };
    // For GetField
    struct {
      VirtualRegisterNum object;
      size_t field_index;
    };
    // For AllocDatatype
    struct {
      size_t constructor_tag;
      size_t num_fields;
      VirtualRegisterNum* datatype_fields;
    };
    // For AllocClosure
    struct {
      size_t clo_index;
      size_t num_freevar;
      VirtualRegisterNum* free_vars;
    };
  };

  Instruction();
  Instruction(const Instruction& instr);
  ~Instruction();

  friend std::ostream& operator<<(std::ostream& os, const Instruction&);
};

// Helpers to build instructions.
Instruction Phi(VirtualRegisterNum cond, VirtualRegisterNum op1, VirtualRegisterNum op2, VirtualRegisterNum dst);
Instruction Ret(VirtualRegisterNum result);
Instruction InvokePacked(size_t packed_index, size_t arity, size_t output_size, std::vector<VirtualRegisterNum> args);
Instruction AllocTensor(const std::vector<int64_t>& shape, DLDataType dtype, VirtualRegisterNum dst);
Instruction AllocDatatype(size_t tag, size_t num_fields, std::vector<VirtualRegisterNum> fields, VirtualRegisterNum dst);
Instruction AllocClosure(size_t func_index, size_t num_freevar, std::vector<VirtualRegisterNum> free_vars, VirtualRegisterNum dst);
Instruction GetField(VirtualRegisterNum object, size_t field_index, VirtualRegisterNum dst);
Instruction If(VirtualRegisterNum cond, size_t true_branch, size_t false_branch);
Instruction Goto(size_t pc_offset);
Instruction Invoke(size_t func_index, std::vector<VirtualRegisterNum> args, VirtualRegisterNum dst);
Instruction InvokeClosure(VirtualRegisterNum closure, std::vector<VirtualRegisterNum> args, VirtualRegisterNum dst);
Instruction LoadConst(size_t const_index, VirtualRegisterNum dst);
Instruction Move(VirtualRegisterNum src, VirtualRegisterNum dst);

struct VMFunction {
  std::string name;
  size_t params;
  std::vector<Instruction> instructions;
  size_t register_file_size;

  // register num -> memory slot
  std::unordered_map<size_t, size_t> register_file_map;

  VMFunction(std::string name, size_t params, std::vector<Instruction> instructions, size_t register_file_size,
             std::unordered_map<size_t, size_t> register_file_map)
    : name(name), params(params), instructions(instructions), register_file_size(register_file_size),
      register_file_map(register_file_map) {}

  VMFunction() {}

  friend std::ostream& operator<<(std::ostream& os, const VMFunction&);
};

void VMFunctionPrint(const VMFunction& vm_func);

/*! \brief A representation of a stack frame.
 *
 * We store the current frame's information on the call stack (frames)
 * when we finish execution we restore the virtual machine state.
 */
struct VMFrame {
    size_t pc;
    size_t func_index;
    size_t args;
    const Instruction* code;

    // Points to the offset of this function's register file start inside VM's register stack
    size_t register_stack_start;

    // map from virtual register to memory slot
    std::unordered_map<VirtualRegisterNum, SlotNum> register_file_map;

    VMFrame(size_t pc, size_t func_index, size_t args, const Instruction* code, size_t register_stack_start,
            std::unordered_map<size_t, size_t> register_file_map)
      : pc(pc), func_index(func_index), args(args), code(code), register_stack_start(register_stack_start),
        register_file_map(register_file_map) {}
};

struct VirtualMachine {
    // TODO(@jroesch):
    std::vector<PackedFunc> packed_funcs;
    std::vector<VMFunction> functions;
    std::vector<VMFrame> frames;
    std::vector<Object> constants;

    // Frame State
    size_t func_index;
    const Instruction* code;
    size_t pc;
    // Top position of register stack
    size_t top_stack;
    // Special register to save function call return value
    Object return_register;
    // Global memory used by all functions' registers
    std::vector<Object> register_stack;

    std::vector<TVMContext> ctxs;

    // Interface debugging.
    std::unordered_map<GlobalVar, size_t, NodeHash, NodeEqual> global_map;
    std::unordered_map<size_t, Constructor> tag_index_map;

    void PushFrame(size_t arg_count, size_t ret_pc, const VMFunction& vm_func);
    size_t PopFrame();
    void InvokeGlobal(const VMFunction& func, const std::vector<Object>& args);
    void Run();

    // Find a register's offset inside the register_stack. 
    size_t LookupRegister(VirtualRegisterNum r);
    void WriteRegister(VirtualRegisterNum r, Object v);
    Object ReadRegister(VirtualRegisterNum r);

    Object Invoke(const VMFunction& func, const std::vector<Object>& args);
    Object Invoke(const GlobalVar& global, const std::vector<Object>& args);

    VirtualMachine() :
      functions(), frames(),
      func_index(0), code(nullptr), pc(0) {}

    void Init(const std::vector<TVMContext>& ctxs);

    static VirtualMachine FromModule(const Module& module,
                                     const std::vector<TVMContext>& ctxs);
};

bool IsClosure(const Function& func);
Module LambdaLift(const Module& module);
Module InlinePrimitives(const Module& module);

VirtualMachine CompileModule(const Module& mod);

}  // namespace vm
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_RUNTIME_H_
