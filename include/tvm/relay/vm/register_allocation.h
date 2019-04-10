/*!
 *  Copyright (c) 2019 by Contributors
 * \file tvm/relay/vm/register_allocation.h
 * \brief register allocation pass definition
 * 
 * Implemented the algorithm in:
 * Poletto, M., & Sarkar, V. (1999). Linear scan register allocation. 
 * ACM Transactions on Programming Languages and Systems (TOPLAS), 21(5), 895-913.
 * 
 * We don't consider spilling and register coalescing right now.
 */
#ifndef TVM_RELAY_VM_REGISTER_ALLOCATION_H_
#define TVM_RELAY_VM_REGISTER_ALLOCATION_H_

#include <memory>
#include <unordered_map>
#include <vector>

namespace tvm {
namespace relay {
namespace vm {

using VirtualRegisterNum = size_t;
using SlotNum = size_t;

class LiveInterval {
public:
  LiveInterval(VirtualRegisterNum reg, size_t line_start, size_t line_end)
  : reg(reg), line_start(line_start), line_end(line_end) {}

  VirtualRegisterNum reg;
  size_t line_start{INT_MAX};
  size_t line_end{0};

  void UpdateLineStart(size_t line) {
    line_start = std::min(line_start, line);
  }

  void UpdateLineEnd(size_t line) {
    line_end = std::max(line_end, line);
  }
};

std::pair<std::unordered_map<VirtualRegisterNum, SlotNum>, SlotNum>
RegisterAllocation(std::vector<LiveInterval> live_intervals);

}  // namespace vm
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_VM_REGISTER_ALLOCATION_H_
