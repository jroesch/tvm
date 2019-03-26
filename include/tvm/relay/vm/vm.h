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

namespace tvm {
namespace relay {
namespace vm {

using namespace tvm::runtime;

enum struct Opcode {
  Push,
  Pop,
  Ret,
  Invoke,
  InvokeClosure,
  InvokePacked,
  AllocTensor,
  AllocDatatype,
  AllocClosure,
  GetField,
  If,
  LoadConst,
  Goto,
  Move,
};

struct Instruction {
  struct TensorInfo {
      int64_t* shape;
      size_t ndim;
      DLDataType dtype;
  };

  Opcode op;
  union {
    size_t stack_index;
    TensorInfo tensor_info;
    struct {
      size_t packed_index;
      size_t arity;
      size_t output_size;
    };
    struct {
      size_t true_offset;
      size_t false_offset;
    };
    struct {
      size_t func_index;
    };
    struct {
      size_t const_index;
    };
    struct {
      size_t pc_offset;
    };
    struct {
      size_t object_offset;
      size_t field_index;
    };
    struct {
      size_t constructor_tag;
      size_t num_fields;
    };
    struct {
      size_t clo_index;
      size_t num_freevar;
    };
    struct {
      size_t pop_count;
    };
    struct {
      size_t source;
      size_t dest;
    };
  };

  Instruction();
  Instruction(const Instruction& instr);
  ~Instruction();

  friend std::ostream& operator<<(std::ostream& os, const Instruction&);
};

// Helpers to build instructions.
Instruction Push(size_t stack_index);
Instruction Pop(size_t pop_count);
Instruction Ret();
Instruction InvokePacked(size_t packed_index, size_t arity, size_t output_size);
Instruction AllocTensor(const std::vector<int64_t>& shape, DLDataType dtype);
Instruction AllocDatatype(size_t tag, size_t num_fields);
Instruction AllocClosure(size_t func_index, size_t num_freevar);
Instruction GetField(size_t object_offset, size_t field_index);
Instruction If(size_t true_branch, size_t false_branch);
Instruction Goto(size_t pc_offset);
Instruction Invoke(size_t func_index);
Instruction InvokeClosure();
Instruction LoadConst(size_t const_index);
Instruction Move(size_t source, size_t dest);

struct VMFunction {
  std::string name;
  size_t params;
  std::vector<Instruction> instructions;

  VMFunction(std::string name, size_t params, std::vector<Instruction> instructions)
    : name(name), params(params), instructions(instructions) {}

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
    size_t bp;
    size_t sp;
    size_t func_index;
    size_t args;
    const Instruction* code;

    VMFrame(size_t pc, size_t bp, size_t sp, size_t func_index, size_t args, const Instruction* code)
      : pc(pc), bp(bp), sp(sp), func_index(func_index), args(args), code(code) {}
};

struct VirtualMachine {
    // TODO(@jroesch):
    std::vector<PackedFunc> packed_funcs;
    std::vector<VMFunction> functions;
    std::vector<VMFrame> frames;
    std::vector<Object> stack;
    std::vector<Object> constants;

    // Frame State
    size_t func_index;
    const Instruction* code;
    size_t pc;
    size_t bp;

    std::vector<TVMContext> ctxs;

    // Interface debugging.
    std::unordered_map<GlobalVar, size_t, NodeHash, NodeEqual> global_map;
    std::unordered_map<size_t, Constructor> tag_index_map;

    void PushFrame(size_t arg_count, size_t ret_pc, size_t sp, const VMFunction& vm_func);
    size_t PopFrame();
    void InvokeGlobal(const VMFunction& func);
    void Run();

    Object Invoke(const VMFunction& func, const std::vector<Object>& args);
    Object Invoke(const GlobalVar& global, const std::vector<Object>& args);

    // Ignore the method that dumps register info at compile-time if debugging
    // mode is not enabled.
    template <typename T = EnableRelayDebug>
    typename std::enable_if<T::value, void>::type
    DumpRegister();

    template <typename T = EnableRelayDebug>
    typename std::enable_if<!T::value, void>::type
    DumpRegister() {}

    // Ignore the method that dumps stack info at compile-time if debugging
    // mode is not enabled.
    template <typename T = EnableRelayDebug>
    typename std::enable_if<T::value, void>::type DumpStack();

    template <typename T = EnableRelayDebug>
    typename std::enable_if<!T::value, void>::type DumpStack() {}

    VirtualMachine() :
      functions(), frames(), stack(),
      func_index(0), code(nullptr), pc(0), bp(0) {}

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
