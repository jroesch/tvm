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
 * \file src/relay/ir/expr.cc
 * \brief The expression AST nodes of Relay.
 */
#include <tvm/ir/module.h>
#include <tvm/relax/expr.h>

namespace tvm {
namespace relax {

void BinaryBroadcastShapeFn(int lhs_rank, int rhs_rank, int out_rank, void *lhs, void *rhs, void *out) {
  std::cout << lhs_rank << std::endl;
  std::cout << rhs_rank << std::endl;
  std::cout << out_rank << std::endl;
  int32_t* lhs_int = static_cast<int32_t*>(lhs);
  int32_t* rhs_int = static_cast<int32_t*>(rhs);
  int32_t* out_int = static_cast<int32_t*>(out);
  if (lhs_rank == rhs_rank) {
    CHECK_EQ(lhs_rank, out_rank);
    for (int i = 0; i < lhs_rank; i++) {
      if (lhs_int[i] == rhs_int[i]) {
        out_int[i] = lhs_int[i];
      } else {
        LOG(FATAL) << "fix me " << lhs_int[i] << "\n" << rhs_int[i];
      }
    }
  } else {
    LOG(FATAL) << "yolo";
  }
}

TVM_REGISTER_GLOBAL("relax.binary_broadcast_shape_fn")
.set_body_typed(BinaryBroadcastShapeFn);

template<typename T>
void LayoutTensorData(std::ostream& ostream, int64_t* shape, int dim, int current_dim, size_t index, T* data) {
    if (current_dim == dim) {
        ostream << data[index];
    } else {

    }
}

void GetRank(DLTensor* tensor) {
    std::cout << "Tensor(shape=";

    std::cout << "(";
    for (int i = 0; i < tensor->ndim; i++) {
        std::cout << tensor->shape[i] << ",";
    }
    std::cout << "), dtype=";

    switch (static_cast<DLDataTypeCode>(tensor->dtype.code)) {
        case kDLInt:
            std::cout << "int";
            break;
        case kDLUInt:
            std::cout << "uint";
            break;
        case kDLFloat:
            std::cout << "float";
            break;
        default:
            LOG(FATAL) << "foo";
    }

    std::cout << static_cast<int>(tensor->dtype.bits);
    std::cout << ", data=" << std::endl;

    std::cout << ")";
    std::cout << std::endl;
}

TVM_REGISTER_GLOBAL("relax.get_rank")
.set_body_typed(GetRank);

}  // namespace relay
}  // namespace tvm
