/*!
 *  Copyright (c) 2018 by Contributors
 * \file type_subst.cc
 * \brief Function for substituting a concrete type in place of a type ID
 */
#include <tvm/relay/pass.h>
#include <tvm/relay/expr_visitor.h>
#include <unordered_set>

namespace tvm {
namespace relay {

struct ShadowDetected { };

struct DetectShadow : ExprVisitor {
  std::unordered_set<LocalVar> s;
};

bool LocalVarWellFormed(const Expr & e) {
  try {
    DetectShadow()(e);
    return true;
  } catch (const ShadowDetected &) {
    return false;
  }
}

}  // namespace relay
}  // namespace tvm
