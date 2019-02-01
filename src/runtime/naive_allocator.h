/*!
 *  Copyright (c) 2019 by Contributors
 * \file runtime/naive_allocator.h
 */
#ifndef TVM_RUNTIME_NAIVE_ALLOCATOR_H_
#define TVM_RUNTIME_NAIVE_ALLOCATOR_H_

#include <atomic>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/memory_manager.h>

namespace tvm {
namespace runtime {

class NaiveAllocator final : public Allocator {
 public:
  NaiveAllocator(TVMContext ctx) : Allocator(ctx), used_memory_(0) {}
  
  Buffer Alloc(size_t nbytes, size_t alignment, TVMType type_hint) override {
    Buffer buf;
    buf.ctx = ctx_;
    buf.size = nbytes;
    buf.data = DeviceAPI::Get(ctx_)->AllocDataSpace(
        ctx_, nbytes, alignment, type_hint);
    used_memory_.fetch_add(nbytes, std::memory_order_relaxed);
    LOG(INFO) << "allocate " << nbytes << " B, used memory " << used_memory_ << " B";
    return buf;
  }

  void Free(const Buffer& buffer) override {
    DeviceAPI::Get(ctx_)->FreeDataSpace(buffer.ctx, buffer.data);
    used_memory_.fetch_sub(buffer.size, std::memory_order_relaxed);
    LOG(INFO) << "free " << buffer.size << " B, used memory " << used_memory_ << " B";
  }

  size_t UsedMemory() override {
    return used_memory_.load(std::memory_order_relaxed);
  }

 private:
  std::atomic<size_t> used_memory_;
};

}  // namespace runtime
}  // namespace tvm

#endif // TVM_RUNTIME_NAIVE_ALLOCATOR_H_
