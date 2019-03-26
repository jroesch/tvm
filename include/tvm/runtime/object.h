/*!
 *  Copyright (c) 2019 by Contributors
 * \file tvm/runtime/object.h
 * \brief A managed object in the TVM runtime.
 */
#ifndef TVM_RUNTIME_OBJECT_H_
#define TVM_RUNTIME_OBJECT_H_

#include <tvm/runtime/ndarray.h>

namespace tvm {
namespace runtime {

enum struct ObjectTag {
  kTensor,
  kClosure,
  kDatatype,
  kExternalFunc
};

std::ostream& operator<<(std::ostream& os, const ObjectTag&);

// TODO(@jroesch): Use intrusive pointer.
struct ObjectCell {
  ObjectTag tag;
  ObjectCell(ObjectTag tag) : tag(tag) {}
  ObjectCell() {}
  virtual ~ObjectCell() {}
};

/*!
 * \brief A managed object in the TVM runtime.
 *
 * For example a tuple, list, closure, and so on.
 *
 * Maintains a reference count for the object.
 */
class Object {
public:
  std::shared_ptr<ObjectCell> ptr;
  Object(std::shared_ptr<ObjectCell> ptr) : ptr(ptr) {}
  Object() : ptr() {}
  Object(const Object& obj) : ptr(obj.ptr) {}
  ObjectCell* operator->() {
    return this->ptr.operator->();
  }
};

struct TensorCell : public ObjectCell {
  NDArray data;
  TensorCell(const NDArray& data)
    : ObjectCell(ObjectTag::kTensor), data(data) {}
};

struct DatatypeCell : public ObjectCell {
  size_t tag;
  std::vector<Object> fields;

  DatatypeCell(size_t tag, const std::vector<Object>& fields)
    : ObjectCell(ObjectTag::kDatatype), tag(tag), fields(fields) {}
};

struct ClosureCell : public ObjectCell {
  size_t func_index;
  std::vector<Object> free_vars;

  ClosureCell(size_t func_index, const std::vector<Object>& free_vars)
    : ObjectCell(ObjectTag::kClosure), func_index(func_index), free_vars(free_vars) {}
};

Object TensorObj(const NDArray& data);
Object DatatypeObj(size_t tag, const std::vector<Object>& fields);
Object TupleObj(const std::vector<Object>& fields);
Object ClosureObj(size_t func_index, std::vector<Object> free_vars);
NDArray ToNDArray(const Object& obj);

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_OBJECT_H_
