/*!
 * Copyright (c) 2019 by Contributors
 * \file tvm/relay/vm/register_allocation.cc
 */

#include <tvm/relay/vm/register_allocation.h>

#include <queue>
#include <unordered_set>
#include <vector>
#include <iostream>

namespace tvm {
namespace relay {
namespace vm {

std::unordered_map<VirtualRegisterNum, SlotNum>
RegisterAllocation(std::vector<LiveInterval> live_intervals) {
  std::sort(live_intervals.begin(), live_intervals.end(), [](const LiveInterval& a, const LiveInterval& b) {
    if (a.line_start != a.line_start) {
      return a.line_start < b.line_start;
    }
    return a.line_end < b.line_end;
  });

  std::unordered_map<VirtualRegisterNum, SlotNum> register_file_map;
  std::queue<LiveInterval> active;
  std::unordered_set<SlotNum> used;
  std::unordered_set<SlotNum> freed;  
  SlotNum next_slot = 0;

  for (auto next_interval : live_intervals) {
    while (!active.empty() && (active.front().line_end < next_interval.line_start)) {      
      used.erase(register_file_map.at(active.front().reg));
      freed.insert(register_file_map.at(active.front().reg));
      active.pop();
    }
    active.push(next_interval);
    if (freed.empty()) {
      auto assign = next_slot++;
      used.insert(assign);
      register_file_map[next_interval.reg] = assign;
    } else {
      auto assign = *freed.begin();
      freed.erase(assign);
      register_file_map[next_interval.reg] = assign;
    }
  }
  return register_file_map;
}

}  // namespace vm
}  // namespace relay
}  // namespace tvm
