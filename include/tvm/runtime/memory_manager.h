/*!
 *  Copyright (c) 2019 by Contributors
 * \file tvm/runtime/memory_manager.h
 * \brief Abstract device memory management API
 */
#ifndef TVM_RUNTIME_MEMORY_MANAGER_H_
#define TVM_RUNTIME_MEMORY_MANAGER_H_

#include <memory>
#include <mutex>
#include <vector>
#include <unordered_map>
#include "c_runtime_api.h"

namespace std {
template<>
struct hash<TVMContext> {
  std::size_t operator()(const TVMContext& ctx) const {
    return ((ctx.device_id << 8) | ctx.device_type);
  }
};

template<>
struct equal_to<TVMContext> {
  bool operator()(const TVMContext& lhs, const TVMContext& rhs) const {
    return (lhs.device_type == rhs.device_type &&
            lhs.device_id == rhs.device_id);
  }
};

} // namespace std

namespace tvm {
namespace runtime {

struct Buffer {
  // data pointer
  void* data{nullptr};
  // Buffer size in bytes
  size_t size{0};
  // TVM Context
  TVMContext ctx;
};

class Allocator {
 public:
  Allocator(TVMContext ctx) : ctx_(ctx) {}
  
  virtual Buffer Alloc(size_t nbytes, size_t alignment, TVMType type_hint) = 0;
  virtual void Free(const Buffer& buffer) = 0;
  virtual size_t UsedMemory() = 0;
  virtual ~Allocator() = default;

 protected:
  TVMContext ctx_;
};

class MemoryManager {
 public:
  static MemoryManager* Global();

  Allocator* GetAllocator(TVMContext ctx);
  
 private:
  MemoryManager() {}

 private:
  std::mutex mu_;
  std::unordered_map<TVMContext, std::unique_ptr<Allocator>> allocators_;
};

}  // namespace runtime
}  // namespace tvm

#endif // TVM_RUNTIME_MEMORY_MANAGER_H_
