/*!
 *  Copyright (c) 2018 by Contributors
 * \file type_relations.cc
 * \brief A set of utilities and common functionality
 * for type relations.
 */
#include <tvm/relay/expr.h>
#include <tvm/relay/logging.h>
#include <tvm/relay/op.h>
#include "../pass/incomplete_type.h"

namespace tvm {
namespace relay {

TensorType as_ttype(const Type& t) {
  if (auto tt_node = t.as<TensorTypeNode>()) {
    return GetRef<TensorType>(tt_node);
  } else {
    return TensorType(nullptr);
  }
}

// TODO(@jroesch) what size value do we extract?
int to_int(const tvm::Expr& e) {
  auto imm = e.as<tvm::ir::IntImm>();
  CHECK(imm);
  std::cout << "TYPE: " << imm << imm->type << std::endl;
  return imm->value;
}

Array<Type> IdentityRel(const Array<Type>& types, int num_args) {
  CHECK(types.size() == 2);
  auto t1 = as_ttype(types[0]);
  if (t1 && types[1].as<IncompleteTypeNode>()) {
    return {t1, t1};
  } else {
    return types;
  }
}

static Type ConcreteBroadcast(const TensorType& t1, const TensorType& t2,
                              DataType output_dtype) {
  RELAY_LOG(INFO) << "ConcreteBroadcast: t1=" << t1 << " t2=" << t2
                  << std::endl;
  auto sh1 = t1->shape;
  auto sh2 = t2->shape;
  RELAY_LOG(INFO) << "ConcreteBroadcast: sh1=" << sh1 << " sh2=" << sh2
                  << std::endl;
  if (sh1.size() == 0 && sh2.size() == 0) {
    return TensorTypeNode::make({}, output_dtype);
    // We have non-zero shapes so broadcast rules apply.
  } else {
    auto suffix_len = static_cast<int>(std::min(sh1.size(), sh2.size()));
    auto full_len = static_cast<int>(std::max(sh1.size(), sh2.size()));

    std::cout << "Length" << suffix_len << full_len
              << (full_len - suffix_len - 1) << std::endl;
    auto lower_bound = full_len - suffix_len - 1;

    for (int64_t i = full_len - 1; i > lower_bound; i--) {
      std::cout << "Index i=" << i << std::endl;
      auto dim1 = to_int(sh1[i]);
      auto dim2 = to_int(sh2[i]);
      if (dim1 != dim2) {
        CHECK(false);
      }
    }

    Array<HalideIR::Expr> larger;
    Array<HalideIR::Expr> smaller;

    for (int i = 0; i < (full_len - suffix_len); i++) {
      smaller.push_back(tvm::ir::IntImm::make(HalideIR::Int(64), 1));
    }

    if (sh1.size() < sh2.size()) {
      for (auto sh : sh1) {
        smaller.push_back(sh);
      }
      larger = sh2;
    } else if (sh1.size() > sh2.size()) {
      for (auto sh : sh1) {
        larger.push_back(sh);
      }
      smaller = sh2;
    } else {
      larger = sh1;
      smaller = sh2;
    }

    CHECK(larger.size() == smaller.size());

    Array<HalideIR::Expr> out_shape;
    for (int i = 0; i < smaller.size(); i++) {
      auto left = smaller[i].as<tvm::ir::IntImm>();
      auto right = larger[i].as<tvm::ir::IntImm>();
      CHECK(left);
      CHECK(right);
      int64_t dim = std::max(left->value, right->value);
      out_shape.push_back(tvm::ir::IntImm::make(HalideIR::Int(64), dim));
    }

    return TensorTypeNode::make(out_shape, output_dtype);
  }
}

Array<Type> BroadcastRel(const Array<Type>& types, int num_args) {
  CHECK(types.size() == 3);
  if (auto t1 = as_ttype(types[0])) {
    if (auto t2 = as_ttype(types[1])) {
      std::cout << t1->dtype << t2->dtype << std::endl;
      CHECK(t1->dtype == t2->dtype);
      return {t1, t2, ConcreteBroadcast(t1, t2, t1->dtype)};
    }
  }

  return types;
}

/* A relation which specifies broadcasting rules for operations which
   compute boolean results.
*/
Array<Type> BroadcastCompRel(const Array<Type>& types, int num_args) {
  CHECK(types.size() == 3);
  if (auto t1 = as_ttype(types[0])) {
    if (auto t2 = as_ttype(types[1])) {
      return {t1, t2, ConcreteBroadcast(t1, t2, HalideIR::Bool())};
    }
  }

  return types;
}

}  // namespace relay
}  // namespace tvm
