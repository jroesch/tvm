/*!
 *  Copyright (c) 2018 by Contributors
 * \file tvm/relay/runtime/runtime.h
 * \brief Abstract device memory management API
 */
#ifndef TVM_RELAY_RUNTIME_H_
#define TVM_RELAY_RUNTIME_H_

#include <tvm/relay/expr_functor.h>
#include<vector>

namespace tvm {
namespace relay {
namespace vm {

using runtime::NDArray;

enum struct VMObjectTag {
  kClosure,
  kDatatype,
  kTensor,
  kExternalFunc,
};


// TODO(@jroesch): Eventually inline cell.
// We can also use pointer tagging scheme ala
// https://github.com/leanprover/lean/blob/master/src/library/vm/vm.h#L51
// https://www.microsoft.com/en-us/research/wp-content/uploads/2007/10/ptr-tagging.pdf?from=http%3A%2F%2Fresearch.microsoft.com%2Fen-us%2Fum%2Fpeople%2Fsimonpj%2Fpapers%2Fptr-tag%2Fptr-tagging.pdf
struct VMObjectCell {
  VMObjectTag tag;
  VMObjectCell(VMObjectTag tag) : tag(tag) {}
  virtual ~VMObjectCell() {}
};

struct VMTensorCell : public VMObjectCell {
  tvm::runtime::NDArray data;
  VMTensorCell(const tvm::runtime::NDArray data)
    : VMObjectCell(VMObjectTag::kTensor), data(data) {}
};

using VMObject = std::shared_ptr<VMObjectCell>;

VMObject VMTensor(const tvm::runtime::NDArray& data) {
  auto ptr = std::make_shared<VMTensorCell>(data);
  return std::dynamic_pointer_cast<VMObjectCell>(ptr);
}

inline NDArray ToNDArray(const VMObject& obj) {
  CHECK(obj.get());
  CHECK(obj->tag == VMObjectTag::kTensor);
  std::shared_ptr<VMTensorCell> o = std::dynamic_pointer_cast<VMTensorCell>(obj);
  return o->data;
}

enum struct Opcode {
  Push,
  Ret,
  Invoke,
  InvokePacked,
  AllocTensor,
  If,
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
    };
    struct {
      size_t true_offset;
      size_t false_offset;
    };
    struct {
      size_t func_index;
    };
  };

  Instruction();
  Instruction(const Instruction& instr);
  ~Instruction();
};

Instruction Push(size_t stack_index);
Instruction Ret();
Instruction InvokePacked(size_t stack_index);
Instruction AllocTensor(std::vector<int64_t> shape, std::string dtype);


struct VMFunction {
  size_t params;
  std::vector<Instruction> instructions;
  VMFunction(size_t params, std::vector<Instruction> instructions)
    : params(params), instructions(instructions) {}
  VMFunction() {}
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
    size_t func_index;
    size_t args;
    const Instruction* code;

    VMFrame(size_t pc, size_t bp, size_t func_index, size_t args, const Instruction* code)
      : pc(pc), bp(bp), func_index(func_index), args(args), code(code) {}
};

struct VirtualMachine {
    // TODO(@jroesch):
    std::vector<PackedFunc> packed_funcs;
    std::vector<VMFunction> functions;
    std::vector<VMFrame> frames;
    std::vector<VMObject> stack;
    std::vector<VMObject> constants;

    size_t func_index;
    const Instruction* code;
    size_t pc;
    size_t bp;

    void PushFrame(size_t arg_count, size_t ret_pc, const VMFunction& vm_func);
    size_t PopFrame();
    void InvokeGlobal(const VMFunction& func, const std::vector<VMObject>& args);
    void Run();

    VMObject Invoke(const VMFunction& func, const std::vector<VMObject>& args);

    VirtualMachine() :
      functions(), frames(), stack(),
      func_index(0), code(nullptr), pc(0), bp(0) {}
};

VirtualMachine CompileModule(const Module& mod);

}  // namespace vm
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_RUNTIME_H_
