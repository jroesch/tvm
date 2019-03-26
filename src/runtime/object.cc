/*!
 *  Copyright (c) 2017 by Contributors
 * \file object.cc
 * \brief TVM runtime object used by VM.
 */

#include <tvm/logging.h>
#include <tvm/runtime/object.h>
#include <iostream>

namespace tvm {
namespace runtime {

std::ostream& operator<<(std::ostream& os, const ObjectTag& tag) {
  switch (tag) {
    case ObjectTag::kClosure:
      os << "Closure";
      break;
    case ObjectTag::kDatatype:
      os << "Datatype";
      break;
    case ObjectTag::kTensor:
      os << "Tensor";
      break;
    case ObjectTag::kExternalFunc:
      os << "ExternalFunction";
      break;
    default:
      LOG(FATAL) << "Invalid object tag: found " << static_cast<int>(tag);
  }
  return os;
}

Object TensorObj(const NDArray& data) {
  auto ptr = std::make_shared<TensorCell>(data);
  return std::dynamic_pointer_cast<ObjectCell>(ptr);
}

Object DatatypeObj(size_t tag, const std::vector<Object>& fields) {
  auto ptr = std::make_shared<DatatypeCell>(tag, fields);
  return std::dynamic_pointer_cast<ObjectCell>(ptr);
}

Object TupleObj(const std::vector<Object>& fields) {
  return DatatypeObj(0, fields);
}

Object ClosureObj(size_t func_index, std::vector<Object> free_vars) {
  auto ptr = std::make_shared<ClosureCell>(func_index, free_vars);
  return std::dynamic_pointer_cast<ObjectCell>(ptr);
}

NDArray ToNDArray(const Object& obj) {
  CHECK(obj.ptr.get());
  CHECK(obj.ptr->tag == ObjectTag::kTensor) << "Expected tensor, found " << obj.ptr->tag;
  std::shared_ptr<TensorCell> o = std::dynamic_pointer_cast<TensorCell>(obj.ptr);
  return o->data;
}

} // namespace runtime
} // namespace tvm
