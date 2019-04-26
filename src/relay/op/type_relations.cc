/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 *  Copyright (c) 2018 by Contributors
 * \file type_relations.cc
 * \brief A set of utilities and common functionality
 * for type relations.
 */
#include <tvm/relay/expr.h>
#include <tvm/relay/logging.h>
#include <tvm/relay/op.h>
#include <tvm/ir_pass.h>
#include <numeric>
#include "./type_relations.h"

namespace tvm {
namespace relay {

TensorType ToTensorType(const Type& t) {
  if (const auto* tt_node = t.as<TensorTypeNode>()) {
    return GetRef<TensorType>(tt_node);
  } else {
    return TensorType(nullptr);
  }
}

// using TypeRelation =
//   std::function<bool(const Array<Type>& types,
//                      int num_inputs,
//                      const Attrs& attrs,
//                      const TypeReporter& reporter)>;

// struct SymDim {

// };

// struct SymShape {
//   Array<SymDim> shape;
//   std::vector<tvm::Expr> matched_shape;
//   DataType dtype;

//   SymShape() : shape(Array<SymDim>()) {}

//   bool Match(const Type& type) {
//     auto is_inc = type.as<IncompleteTypeNode>();
//     if (is_inc) {
//       return false;
//     }

//     auto is_tensor = type.as<TensorTypeNode>();

//     if (!is_tensor) {
//       LOG(FATAL) << "must be tensor";
//     }

//     // matches any
//     if (!shape.defined()) {
//       for (auto sh : is_tensor->shape) {
//         matched_shape.push_back(sh);
//       }
//       dtype = is_tensor->dtype;
//       return true;
//     }

//     LOG(FATAL) << "NYI";
//   }
// };

// bool ShapeRelation(
//   const Array<Type>& types,
//   int num_inputs,
//   const Attrs& attrs,
//   const TypeReporter& reporter,
//   std::function<SymShape(const Array<Type>& inputs)> body) {
//     CHECK(num_inputs == types.size() - 1);
//     Array<Type> inputs = Array<Type>(types.begin(), types.end());
//     auto out_shape = body(inputs);
// }

// bool MatchShapes(const Array<Type>& types, const std::vector<SymShape>& pattern) {
//   CHECK(types.size() == pattern.size());
//   bool matches = false;
//   for (size_t i = 0; i < types.size(); i++) {
//     SymShape& sh = const_cast<SymShape&>(pattern[i]);
//     matches = matches || sh.Match(types[i]);
//   }
//   return matches;
// }

bool IdentityRel(const Array<Type>& types,
                 int num_inputs,
                 const Attrs& attrs,
                 const TypeReporter& reporter) {
  for (size_t i = 1; i < types.size(); ++i) {
    reporter->Assign(types[i], types[0]);
  }
  return true;
  // return ShapeRelation(types, num_inputs, attrs, reporter,
  //   [&](const Array<Type>& inputs) {
  //   SymShape s;
  //   if (MatchShapes(inputs, {s})) {
  //     return s;
  //   }
  // });
}

bool EqualCheck(const IndexExpr& lhs,
                const IndexExpr& rhs) {
  IndexExpr diff = lhs - rhs;
  if (const int64_t* pdiff = as_const_int(diff)) {
    return pdiff[0] == 0;
  }
  // symbolic
  diff = tvm::ir::CanonicalSimplify(diff);
  if (const int64_t* pdiff = as_const_int(diff)) {
    return pdiff[0] == 0;
  }
  return false;
}

bool EqualConstInt(const IndexExpr& lhs, int64_t value) {
  if (const int64_t* pvalue = as_const_int(lhs)) {
    return pvalue[0] == value;
  }
  return false;
}


Type ConcreteBroadcast(const TensorType& t1,
                       const TensorType& t2,
                       DataType output_dtype) {
  std::vector<IndexExpr> oshape;
  size_t ndim1 = t1->shape.size();
  size_t ndim2 = t2->shape.size();
  size_t i = 1;
  for (; i <= std::min(ndim1, ndim2); ++i) {
    IndexExpr s1 = t1->shape[ndim1 - i];
    IndexExpr s2 = t2->shape[ndim2 - i];
    if (EqualCheck(s1, s2)) {
      oshape.push_back(s1);
    } else if (EqualConstInt(s1, 1)) {
      oshape.push_back(s2);
    } else if (EqualConstInt(s2, 1)) {
      oshape.push_back(s1);
    } else if (s1.same_as(Any())) {
      LOG(WARNING) << "Assert any == 1 || any == " << s2 << " in broadcast of "
                   << t1 << " and " << t2;
      // TODO: Assert Any!!! s1 == 1 || s1 == s2
      oshape.push_back(s2);
    } else if (s2.same_as(Any())) {
      LOG(WARNING) << "Assert any == 1 || any == " << s1 << " in broadcast of "
                   << t1 << " and " << t2;
      // TODO: Assert Any!!! s2 == 1 || s2 == s1
      oshape.push_back(s1);
    } else {
      RELAY_ERROR(
          "Incompatible broadcast type "
              << t1 << " and " << t2).Raise();
    }
  }

  size_t max_ndim = std::max(ndim1, ndim2);
  auto& rshape = (ndim1 > ndim2) ? t1->shape : t2->shape;
  for (; i <= max_ndim; ++i) {
    oshape.push_back(rshape[max_ndim - i]);
  }
  return TensorTypeNode::make(Array<IndexExpr>(
      oshape.rbegin(), oshape.rend()), output_dtype);
}

bool BroadcastRel(const Array<Type>& types,
                  int num_inputs,
                  const Attrs& attrs,
                  const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 3);
  RELAY_LOG(INFO) << "In1:" << types[0] << ",In2:" << types[1]
                  << ",Out:" << types[2] << std::endl;
  if (auto t0 = ToTensorType(types[0])) {
    if (auto t1 = ToTensorType(types[1])) {
      CHECK_EQ(t0->dtype, t1->dtype);
      reporter->Assign(types[2],
        ConcreteBroadcast(t0, t1, t0->dtype));
      return true;
    }
  }
  return false;
}

bool BroadcastCompRel(const Array<Type>& types,
                      int num_inputs,
                      const Attrs& attrs,
                      const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 3);
  RELAY_LOG(INFO) << "In1:" << types[0] << ",In2:" << types[1]
                  << ",Out:" << types[2] << std::endl;
  if (auto t0 = ToTensorType(types[0])) {
    if (auto t1 = ToTensorType(types[1])) {
      CHECK_EQ(t0->dtype, t1->dtype);
      reporter->Assign(types[2], ConcreteBroadcast(t0, t1, ::tvm::Bool()));
      return true;
    }
  }
  return false;
}

}  // namespace relay
}  // namespace tvm
