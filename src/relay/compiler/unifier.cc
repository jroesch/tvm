/*!
 *  Copyright (c) 2018 by Contributors
 * \file unifier.cc
 * \brief Data structures for type unification
 */

#include "tvm/relay/ir.h"
#include "tvm/relay/logging.h"
#include "tvm/relay/compiler/alpha_eq.h"
#include "./unifier.h"
#include "./type_visitor.h"
// #include "tvm/relay/typeck/kindchecker.h"
// #include "tvm/relay/typeck/type_subst.h"

namespace tvm {
namespace relay {

using tvm::IRPrinter;
using namespace tvm::runtime;

UnionFind UnionFindNode::make(tvm::Map<IncompleteType, Type> uf_map) {
  std::shared_ptr<UnionFindNode> n = std::make_shared<UnionFindNode>();
  n->uf_map = uf_map;
  return UnionFind(n);
}

void UnionFindNode::insert(const IncompleteType &v) { this->uf_map.Set(v, v); }

void UnionFindNode::debug() {
  for (auto entry : this->uf_map) {
    std::cout << entry.first << " = " << entry.second << std::endl;
  }
}

void UnionFindNode::assertAlphaEq(const Type & l, const Type & r) {
  if (!alpha_eq(l, r)) {
    std::stringstream ss;
    ss << "Incompatible parent types in UF:" << l << " and " << r;
    throw UnionFindError(ss.str());
  }
}

void UnionFindNode::unify(const IncompleteType &v1, const Type &t) {
  RELAY_LOG(INFO) << "UnionFindNode::Unify v1=" << v1 << "t=" << t << std::endl;
  auto parent1 = this->find(v1);

  // if t is a type var, then unify parents
  const IncompleteTypeNode *tvn2 = t.as<IncompleteTypeNode>();
  if (tvn2) {
    auto v2 = GetRef<IncompleteType>(tvn2);
    auto parent2 = this->find(v2);

    // if parents are exactly equal, then we're done
    if (parent1 == parent2) {
      return;
    }

    // if first parent is a type var, then can just set its union find map to
    // second parent
    if (const IncompleteTypeNode *pvn1 = parent1.as<IncompleteTypeNode>()) {
      auto pv1 = GetRef<IncompleteType>(pvn1);
      this->uf_map.Set(pv1, parent2);
      // path compression: can also set v1 directly
      this->uf_map.Set(v1, parent2);
      return;
    }

    // if second parent is a type var but first isn't, can set second type var
    if (const IncompleteTypeNode *pvn2 = parent2.as<IncompleteTypeNode>()) {
      auto pv2 = GetRef<IncompleteType>(pvn2);
      this->uf_map.Set(pv2, parent1);
      // path compression: can also set v2 directly
      this->uf_map.Set(v2, parent1);
      return;
    }

    // if both parents are not type vars themselves, check alpha-equality
    assertAlphaEq(parent1, parent2);
    return;
  }

  // if t is not a type var, then unify with v1's parent if parent is a type
  // var; else, check alpha-equality for compatibility
  if (const IncompleteTypeNode *pvn1 = parent1.as<IncompleteTypeNode>()) {
    auto pv1 = GetRef<IncompleteType>(pvn1);
    this->uf_map.Set(pv1, t);
    // path compression: can also set v1 directly
    this->uf_map.Set(v1, t);
    return;
  }

  assertAlphaEq(parent1, t);
}

Type UnionFindNode::find(const IncompleteType &v) {
  // The node has no mapping, so its representative is just itself.
  if (this->uf_map.find(v) == this->uf_map.end()) {
    return v;
  }

  Type parent = this->uf_map.at(v);

  if (v == parent) {
    return v;
  }

  // if parent is not a type var, then it must be the representative type
  const IncompleteTypeNode *rep = parent.as<IncompleteTypeNode>();
  if (!rep) {
    return parent;
  }

  // otherwise, recurse and perform path compression
  IncompleteType pv = GetRef<IncompleteType>(rep);
  Type higher_up = this->find(pv);
  this->uf_map.Set(v, higher_up);
  return higher_up;
}

TVM_REGISTER_API("relay._make.UnionFind")
    .set_body([](TVMArgs args, TVMRetValue *ret) {
      if (args.size() == 0) {
        *ret = UnionFindNode::make({});
      } else {
        *ret = UnionFindNode::make(args[0]);
      }
    });

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
    .set_dispatch<UnionFindNode>([](const UnionFindNode *node,
                                    tvm::IRPrinter *p) {
      p->stream << "UnionFindNode(" << node->uf_map << ")";
    });

TypeUnifier TypeUnifierNode::make(UnionFind uf) {
  std::shared_ptr<TypeUnifierNode> n = std::make_shared<TypeUnifierNode>();
  n->uf = uf;
  return TypeUnifier(n);
}

void TypeUnifierNode::insert(const IncompleteType &v) { this->uf->insert(v); }

Type TypeUnifierNode::unify(const Type &t1, const Type &t2) {
  RELAY_LOG(INFO) << "TypeUnifierNode::unify: t1=" << t1 << " t2=" << t2
                  << std::endl;

  Type unified = this->VisitType(t1, t2);
  // if (!check_kind(unified)) {
    // throw UnificationError("Invalid kinds in unified type");
  // }
  return unified;
}

struct IncompleteTypeSubst : TypeFVisitor {
  const TypeUnifierNode *unifier;

  IncompleteTypeSubst(const TypeUnifierNode *unifier) : unifier(unifier) {}

  // type var: look it up in the type map and recurse
  Type VisitType_(const IncompleteTypeNode *op) override {
    auto tv = GetRef<IncompleteType>(op);
    auto parent = unifier->uf->find(tv);
    if (parent == tv) {
      return tv;
    }
    return this->VisitType(parent);
  }
};

Type TypeUnifierNode::subst(const Type &t) {
  IncompleteTypeSubst tvsubst(this);
  // normalize first so substitutions in quantifiers will be correct
  Type ret = tvsubst.VisitType(t);
  // if (!check_kind(ret)) {
    // std::stringstream ss;
    // ss << "Invalid Kinds in substituted type!";
    // ss << t << std::endl;
    // ss << ret << std::endl;
    // throw SubstitutionError(ss.str());
  // }
  return ret;
}

Type TypeUnifierNode::VisitType_(const IncompleteTypeNode *t1, const Type rt2) {
  IncompleteType tv1 = GetRef<IncompleteType>(t1);
  RELAY_LOG(INFO) << "VisitType_: IncompleteTypeNode t1=" << t1 << " = " << rt2
                  << std::endl;
  this->uf->unify(tv1, rt2);
  auto rep = this->uf->find(tv1);
  RELAY_LOG(INFO) << "VisitType_: IncompleteTypeNode rep=" << rep << std::endl;
  return rep;
}

Type TypeUnifierNode::VisitType_(const TypeParamNode *t1, const Type rt2) {
  TypeParam ti1 = GetRef<TypeParam>(t1);

  // for typevars, remap and attempt to unify if already defined
  if (const IncompleteTypeNode *tvn2 = rt2.as<IncompleteTypeNode>()) {
    return this->unifyWithIncompleteType(ti1, GetRef<IncompleteType>(tvn2));
  }

  // for other type ids, only check equality
  if (const TypeParamNode *tin2 = rt2.as<TypeParamNode>()) {
    TypeParam ti2 = GetRef<TypeParam>(tin2);

    if (ti1 != ti2) {
      throw UnificationError("Attempting to unify non-matching TypeParams");
    }

    return ti1;
  }

  // cannot unify TypeParam with non-TypeParam
  throw UnificationError("Unable to unify TypeParamNode");
}

Type TypeUnifierNode::VisitType_(const FuncTypeNode *t1, const Type rt2) {
    return rt2;
//   TypeArrow ta1 = GetRef<TypeArrow>(t1);

//   // for typevar, remap if necessary
//   if (const IncompleteTypeNode *tvn2 = rt2.as<IncompleteTypeNode>()) {
//     return this->unifyWithIncompleteType(ta1, GetRef<IncompleteType>(tvn2));
//   }

//   // for other arrow, unify arg and ret types
//   if (const TypeArrowNode *tan2 = rt2.as<TypeArrowNode>()) {
//     TypeArrow ta2 = GetRef<TypeArrow>(tan2);

//     if (ta1->arg_types.size() != ta2->arg_types.size()) {
//       throw UnificationError("unable to unify functions of different arities");
//     }

//     tvm::Array<Type> unified_args;
//     for (size_t i = 0; i < ta1->arg_types.size(); i++) {
//       unified_args.push_back(
//           this->VisitType(ta1->arg_types[i], ta2->arg_types[i]));
//     }

//     Type unified_ret_type = this->VisitType(ta1->ret_type, ta2->ret_type);
//     return TypeArrowNode::make(unified_args, unified_ret_type);
//   }

//   throw UnificationError("Unable to unify TypeArrowNode");
// }

// Type TypeUnifierNode::VisitType_(const TypeQuantifierNode *t1, const Type rt2) {
//   TypeQuantifier tq1 = GetRef<TypeQuantifier>(t1);

//   // for typevars, remap and attempt to unify if already defined
//   if (const IncompleteTypeNode *tvn2 = rt2.as<IncompleteTypeNode>()) {
//     return this->unifyWithIncompleteType(tq1, GetRef<IncompleteType>(tvn2));
//   }

//   // for other quantifiers, attempt to unify bound types after normalizing
//   if (const TypeQuantifierNode *tqn2 = rt2.as<TypeQuantifierNode>()) {
//     TypeQuantifier tq2 = GetRef<TypeQuantifier>(tqn2);
//     TypeParam id1 = tq1->id;
//     TypeParam id2 = tq2->id;

//     if (id1->kind != id2->kind) {
//       throw UnificationError(
//           "Cannot unify quantifiers over ids of different kinds");
//     }

//     TypeParam fresh = TypeParamNode::make(id1->name, id1->kind);

//     auto bt1 = type_subst(tq1->boundType, id1, fresh);
//     auto bt2 = type_subst(tq2->boundType, id2, fresh);

//     Type unified_bound_type = this->VisitType(bt1, bt2);
//     return TypeQuantifierNode::make(fresh, unified_bound_type);
//   }

//   // anything else cannot be unified
//   throw UnificationError("Cannot unify TypeQuantifierNode");
}

Type TypeUnifierNode::VisitType_(const TensorTypeNode *t1, const Type rt2) {
  TensorType tt1 = GetRef<TensorType>(t1);

  // for typevars, remap and attempt to unify if already defined
  if (const IncompleteTypeNode *tvn2 = rt2.as<IncompleteTypeNode>()) {
    return this->unifyWithIncompleteType(tt1, GetRef<IncompleteType>(tvn2));
  }

  if (const TensorTypeNode *ttn2 = rt2.as<TensorTypeNode>()) {
    TensorType tt2 = GetRef<TensorType>(ttn2);

    if (!alpha_eq(tt1, tt2)) {
        throw UnificationError("dtypes do not match");
    }

    RELAY_LOG(INFO) << "Unify Tensor Shape s1=" << tt1->shape
                    << " s2= " << tt2->shape << std::endl;
    try {
      // Type unified_shape = this->VisitType(tt1->shape, tt2->shape);
      return rt2;
    } catch (const UnificationError & err) {
      std::cout << "Need to check constraint " << tt1->shape << " = " << tt2->shape << std::endl;
    }

    // fix me
    return rt2;
    // return TensorTypeNode::make(unified_bt, tt2->shape);
  }

  // nothing else can unify
  throw UnificationError("Cannot unify TensorTypeNode");
}

// Type TypeUnifierNode::VisitType_(const TupleTypeNode *t1, const Type rt2) {
//   TupleType pt1 = GetRef<TupleType>(t1);

//   // for typevar, remap and attempt to unify if already defined
//   if (const IncompleteTypeNode *tvn2 = rt2.as<IncompleteTypeNode>()) {
//     return this->unifyWithIncompleteType(pt1, GetRef<IncompleteType>(tvn2));
//   }

//   // for other product types, unify item by item
//   if (const TupleTypeNode *ptn2 = rt2.as<TupleTypeNode>()) {
//     TupleType pt2 = GetRef<TupleType>(ptn2);

//     std::vector<Type> unified_fields;
//     if (pt1->fields.size() != pt2->fields.size()) {
//       throw UnificationError("Product types are of different dimensions");
//     }

//     for (size_t i = 0U; i < pt1->fields.size(); i++) {
//       Type unified = this->VisitType(pt1->fields[i], pt2->fields[i]);
//       unified_fields.push_back(unified);
//     }

//     return TupleTypeNode::make(unified_fields);
//   }

//   // otherwise cannot unify
//   throw UnificationError("Cannot unify TupleTypeNode");
// }

Type TypeUnifierNode::VisitType_(const TypeFunctionNode *sen1, const Type t2) {
//   ShapeExtension sh_ext1 = GetRef<ShapeExtension>(sen1);

//   if (const IncompleteTypeNode *tvn2 = t2.as<IncompleteTypeNode>()) {
//     return this->unifyWithIncompleteType(sh_ext1, GetRef<IncompleteType>(tvn2));
//   }

//   // will only attempt to unify with binary op with same op
//   if (const ShapeExtensionNode *sen2 = t2.as<ShapeExtensionNode>()) {
//     if (sh_ext1->name != sen2->name) {
//       throw UnificationError(
//           "Cannot unify shape projections of different index");
//     }
//   }

//   return sh_ext1;
    return t2;
}

Type TypeUnifierNode::VisitType_(const TypeCallNode *tcn1, const Type t2) {
  TypeCall ty_call1 = GetRef<TypeCall>(tcn1);

  if (const IncompleteTypeNode *tvn2 = t2.as<IncompleteTypeNode>()) {
    return this->unifyWithIncompleteType(ty_call1, GetRef<IncompleteType>(tvn2));
  }

  if (const TypeCallNode *tcn2 = t2.as<TypeCallNode>()) {
    Type unified_func = this->VisitType(ty_call1->func, tcn2->func);

    // For now, we will only unify if they are equal.
    if (ty_call1->args.size() != tcn2->args.size()) {
      throw UnificationError("Cannot unify calls of different number of arguments");
    }

    // Unify members, if possible
    tvm::Array<Type> new_args;
    for (size_t i = 0U; i < ty_call1->args.size(); i++) {
      Type unified_member = this->VisitType(ty_call1->args[i], tcn2->args[i]);
      new_args.push_back(unified_member);
    }

    return TypeCallNode::make(unified_func, new_args);
  } else {
    throw UnificationError("Cannot unify call with non-call");
  }
}

Type TypeUnifierNode::unifyWithIncompleteType(const Type &t1, const IncompleteType tv2) {
  RELAY_LOG(INFO) << "unifyWithIncompleteType: t1=" << t1 << " t2=" << tv2 << std::endl;
  // Fix unify to return new representative
  this->uf->unify(tv2, t1);
  auto rep = this->uf->find(tv2);
  RELAY_LOG(INFO) << "unifyWithIncompleteType: rep =" << rep << std::endl;
  return rep;
}

TVM_REGISTER_API("relay._make.TypeUnifier")
    .set_body([](TVMArgs args, TVMRetValue *ret) {
      if (args.size() < 3) {
        *ret = TypeUnifierNode::make(UnionFindNode::make({}));
      } else {
        *ret = TypeUnifierNode::make(args[0]);
      }
    });

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
    .set_dispatch<TypeUnifierNode>([](const TypeUnifierNode *node,
                                      tvm::IRPrinter *p) {
      p->stream << "TypeUnifierNode(" << node->uf << ")";
    });

TVM_REGISTER_API("relay._unifier.UnionFind_insert")
    .set_body([](TVMArgs args, TVMRetValue *ret) {
      try {
        UnionFind uf = args[0];
        uf->insert(args[1]);
      } catch (std::exception &e) {
        throw UnionFindError(e.what());
      }
    });

TVM_REGISTER_API("relay._unifier.UnionFind_unify")
    .set_body([](TVMArgs args, TVMRetValue *ret) {
      try {
        UnionFind uf = args[0];
        uf->unify(args[1], args[2]);
      } catch (std::exception &e) {
        throw UnionFindError(e.what());
      }
    });

TVM_REGISTER_API("relay._unifier.UnionFind_find")
    .set_body([](TVMArgs args, TVMRetValue *ret) {
      try {
        UnionFind uf = args[0];
        *ret = uf->find(args[1]);
      } catch (std::exception &e) {
        throw UnionFindError(e.what());
      }
    });

TVM_REGISTER_API("relay._unifier.TypeUnifier_insert")
    .set_body([](TVMArgs args, TVMRetValue *ret) {
      try {
        TypeUnifier unifier = args[0];
        IncompleteType var = args[1];
        unifier->insert(var);
      } catch (std::exception &e) {
        throw UnificationError(e.what());
      }
    });

TVM_REGISTER_API("relay._unifier.TypeUnifier_unify")
    .set_body([](TVMArgs args, TVMRetValue *ret) {
      try {
        TypeUnifier unifier = args[0];
        Type t1 = args[1];
        Type t2 = args[2];
        *ret = unifier->unify(t1, t2);
      } catch (std::exception &e) {
        throw UnificationError(e.what());
      }
    });

TVM_REGISTER_API("relay._unifier.TypeUnifier_subst")
    .set_body([](TVMArgs args, TVMRetValue *ret) {
      try {
        TypeUnifier unifier = args[0];
        Type t = args[1];
        *ret = unifier->subst(t);
      } catch (std::exception &e) {
        throw SubstitutionError(e.what());
      }
    });

}  // namespace relay
}  // namespace tvm
