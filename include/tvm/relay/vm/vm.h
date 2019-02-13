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
#include <tvm/runtime/memory_manager.h>

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

inline std::string VMObjectTagString(VMObjectTag tag) {
  switch (tag) {
    case VMObjectTag::kClosure:
      return "Closure";
    case VMObjectTag::kDatatype:
      return "Datatype";
    case VMObjectTag::kTensor:
      return "Tensor";
    case VMObjectTag::kExternalFunc:
      return "ExternalFunction";
  }
}

// TODO(@jroesch): Eventually inline cell.
// We can also use pointer tagging scheme ala
// https://github.com/leanprover/lean/blob/master/src/library/vm/vm.h#L51
// https://www.microsoft.com/en-us/research/wp-content/uploads/2007/10/ptr-tagging.pdf?from=http%3A%2F%2Fresearch.microsoft.com%2Fen-us%2Fum%2Fpeople%2Fsimonpj%2Fpapers%2Fptr-tag%2Fptr-tagging.pdf
struct VMObjectCell {
  VMObjectTag tag;
  VMObjectCell(VMObjectTag tag) : tag(tag) {}
  VMObjectCell() {}
  virtual ~VMObjectCell() {}
};

struct VMTensorCell : public VMObjectCell {
  tvm::runtime::NDArray data;
  VMTensorCell(const tvm::runtime::NDArray& data)
    : VMObjectCell(VMObjectTag::kTensor), data(data) {}
};

struct VMObject {
  std::shared_ptr<VMObjectCell> ptr;
  VMObject(std::shared_ptr<VMObjectCell> ptr) : ptr(ptr) {}
  VMObject() : ptr() {}
  VMObject(const VMObject& obj) : ptr(obj.ptr) {}
  VMObjectCell* operator->() {
    return this->ptr.operator->();
  }
};

struct VMDatatypeCell : public VMObjectCell {
  size_t tag;
  std::vector<VMObject> fields;

  VMDatatypeCell(size_t tag, const std::vector<VMObject>& fields)
    : VMObjectCell(VMObjectTag::kDatatype), tag(tag), fields(fields) {}
};


inline VMObject VMTensor(const tvm::runtime::NDArray& data) {
  auto ptr = std::make_shared<VMTensorCell>(data);
  return std::dynamic_pointer_cast<VMObjectCell>(ptr);
}

inline VMObject VMDatatype(size_t tag, const std::vector<VMObject>& fields) {
  auto ptr = std::make_shared<VMDatatypeCell>(tag, fields);
  return std::dynamic_pointer_cast<VMObjectCell>(ptr);
}

inline VMObject VMTuple(const std::vector<VMObject>& fields) {
  return VMDatatype(0, fields);
}

inline NDArray ToNDArray(const VMObject& obj) {
  CHECK(obj.ptr.get());
  CHECK(obj.ptr->tag == VMObjectTag::kTensor) << "Expect Tensor, Got " << VMObjectTagString(obj.ptr->tag);
  std::shared_ptr<VMTensorCell> o = std::dynamic_pointer_cast<VMTensorCell>(obj.ptr);
  return o->data;
}

enum struct Opcode {
  Push,
  Ret,
  Invoke,
  InvokePacked,
  AllocTensor,
  AllocDatatype,
  GetField,
  If,
  LoadConst,
  Goto,
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
  };

  Instruction();
  Instruction(const Instruction& instr);
  ~Instruction();
};

Instruction Push(size_t stack_index);
Instruction Ret();
Instruction InvokePacked(size_t stack_index);
Instruction AllocTensor(std::vector<int64_t> shape, std::string dtype);
Instruction GetField(size_t object_offset, size_t field_index);


struct VMFunction {
  std::string name;
  size_t params;
  std::vector<Instruction> instructions;

  VMFunction(std::string name, size_t params, std::vector<Instruction> instructions)
    : name(name), params(params), instructions(instructions) {}

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
    std::vector<VMObject> stack;
    std::vector<VMObject> constants;

    // Frame State
    size_t func_index;
    const Instruction* code;
    size_t pc;
    size_t bp;

    std::vector<TVMContext> ctxs;

    bool debug{false};

    // Interface debugging.
    std::unordered_map<GlobalVar, size_t, NodeHash, NodeEqual> global_map;
    std::unordered_map<size_t, Constructor> tag_index_map;

    void PushFrame(size_t arg_count, size_t ret_pc, size_t sp, const VMFunction& vm_func);
    size_t PopFrame();
    void InvokeGlobal(const VMFunction& func, const std::vector<VMObject>& args);
    void Run();

    VMObject Invoke(const VMFunction& func, const std::vector<VMObject>& args);
    VMObject Invoke(const GlobalVar& global, const std::vector<VMObject>& args);

    void DumpRegister();
    void DumpStack();

    VirtualMachine() :
      functions(), frames(), stack(),
      func_index(0), code(nullptr), pc(0), bp(0) {}

    void Init(const std::vector<TVMContext>& ctxs);

    static VirtualMachine FromModule(const Module& module,
                                     const std::vector<TVMContext>& ctxs);
};

VirtualMachine CompileModule(const Module& mod);

}  // namespace vm
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_RUNTIME_H_
