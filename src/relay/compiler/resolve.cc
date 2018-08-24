/*!
 *  Copyright (c) 2018 by Contributors
 * \file unifier.cc
 * \brief Data structures for type unification
 */

#include "./resolve.h"
#include "./type_visitor.h"
#include "tvm/relay/expr_visitor.h"
#include "tvm/relay/ir.h"

namespace tvm {
namespace relay {

// We should probably generalize the subst code.
struct ResolveTypeType : TypeFVisitor {
  const TypeUnifier &unifier;

  explicit ResolveTypeType(const TypeUnifier &unifier) : unifier(unifier) {}

  Type VisitType(const Type &t) override {
    if (!t.defined()) {
      auto inc_ty = IncompleteTypeNode::make(TypeParamNode::Kind::kType);
      unifier->insert(inc_ty);
      return inc_ty;
    } else {
      return TypeFVisitor::VisitType(t);
    }
  }

  Type VisitType_(const IncompleteTypeNode *op) override {
    return unifier->subst(GetRef<IncompleteType>(op));
  }
};

struct ResolveTypeExpr : ExprFVisitor<> {
  const TypeUnifier &unifier;

  explicit ResolveTypeExpr(const TypeUnifier &unifier) : unifier(unifier) {}

  Expr VisitExpr(const Expr &e) {
    // NB: a bit tricky here.
    //
    // We want to store resolved type without having
    // to re-typecheck the entire term.
    //
    // Since we know that e : T[...] under some holes
    // then it is the case that if we resolve types
    // present in e, then we can type it under T
    // with the wholes filled in.
    //
    // We will visit e like normal building a new
    // term, then resolve e's old type and write
    // it back into the new node.
    auto new_e = ExprFVisitor::VisitExpr(e);
    CHECK(e->checked_type_.defined());
    auto resolved_cty = VisitType(e->checked_type_);
    new_e->checked_type_ = resolved_cty;
    return new_e;
  }

  Type VisitType(const Type &t) {
    return ResolveTypeType(unifier).VisitType(t);
  }
};

Type resolve(const TypeUnifier &unifier, const Type &ty) {
  CHECK(ty.defined());
  return ResolveTypeType(unifier).VisitType(ty);
}

Expr resolve(const TypeUnifier &unifier, const Expr &expr) {
  return ResolveTypeExpr(unifier).VisitExpr(expr);
}

struct FullyResolved : TypeVisitor<> {
  bool incomplete;

  FullyResolved() : incomplete(true) {}

  void VisitType(const Type &t) override {
    if (!t.defined()) {
      incomplete = true;
    } else {
      return TypeVisitor<>::VisitType(t);
    }
  }

  void VisitType_(const IncompleteTypeNode *ty_var) override {
    incomplete = false;
  }
};

bool is_fully_resolved(const Type &t) {
  auto fr = FullyResolved();
  fr.VisitType(t);
  return fr.incomplete;
}

}  // namespace relay
}  // namespace tvm
