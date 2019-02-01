#include <tvm/runtime/memory_manager.h>
#include "naive_allocator.h"
#include "pooled_allocator.h"

namespace tvm {
namespace runtime {

MemoryManager* MemoryManager::Global() {
  static MemoryManager memory_manager;
  return &memory_manager;
}

Allocator* MemoryManager::GetAllocator(TVMContext ctx) {
  std::lock_guard<std::mutex> lock(mu_);
  if (allocators_.find(ctx) == allocators_.end()) {
    LOG(INFO) << "New allocator for " << DeviceName(ctx.device_type) << "("
              << ctx.device_id << ")";
    std::unique_ptr<Allocator> alloc(new NaiveAllocator(ctx));
    //std::unique_ptr<Allocator> alloc(new PooledAllocator(ctx, 128));
    allocators_.emplace(ctx, std::move(alloc));
  }
  return allocators_.at(ctx).get();
}

}  // namespace runtime
}  // namespace tvm
