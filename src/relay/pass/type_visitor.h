/*!
 *  Copyright (c) 2018 by Contributors
 * \file type_visitor.h
 * \brief A wrapper around TypeFunctor for common use cases.
 */
#ifndef TVM_RELAY_PASS_TYPE_VISITOR_H_
#define TVM_RELAY_PASS_TYPE_VISITOR_H_

#include <vector>
#include "./type_functor.h"

namespace tvm {
namespace relay {

/*! \brief A type visitor for vistiors which make use of internal
 * mutable state.
 *
 * We recursively visit each type contained inside the visitor.
 */
template <typename... Args>
struct TypeVisitor : ::tvm::relay::TypeFunctor<void(const Type& n, Args...)> {
  void VisitType_(const TypeParamNode* op, Args... args) override {}

  void VisitType_(const FuncTypeNode* op, Args... args) override {
    for (auto type_param : op->type_params) {
      this->VisitType(type_param, args...);
    }

    for (auto type_cs : op->type_constraints) {
      this->VisitType(type_cs, args...);
    }

    for (auto arg_type : op->arg_types) {
      this->VisitType(arg_type, args...);
    }
    this->VisitType(op->ret_type, args...);
  }

  void VisitType_(const TensorTypeNode* op, Args... args) override {}

  void VisitType_(const TupleTypeNode* op, Args... args) override {
    for (const Type& t : op->fields) {
      this->VisitType(t, args...);
    }
  }

  void VisitType_(const TypeRelationNode* op, Args... args) override {
    for (const Type& t : op->args) {
      this->VisitType(t, args...);
    }
  }

  void VisitType_(const IncompleteTypeNode* op, Args... args) override {}
};

// A functional visitor for rebuilding an AST in place.
struct TypeMutator : TypeFunctor<Type(const Type& n, const Type & self)> {
  virtual Type Mutate(const Type & self) {
    return this->VisitType(self, self);
  }
  Type VisitType_(const TensorTypeNode* op, const Type & self) override {
    // TODO(@jroesch): maybe we should recursively visit
    return self;
  }

  Type VisitType_(const TypeParamNode* op, const Type & self) override {
    return self;
  }

  Type VisitType_(const FuncTypeNode* op, const Type & self) override {
    // TODO(@jroesch): handle poly

    Array<TypeConstraint> type_constraints;
    for (auto type_cs : op->type_constraints) {
      auto new_type_cs = Mutate(type_cs);
      if (const TypeConstraintNode* tin = As<TypeConstraintNode>(new_type_cs)) {
        type_constraints.push_back(GetRef<TypeConstraint>(tin));
      } else {
        CHECK(false) << new_type_cs << std::endl;
      }
    }

    std::vector<Type> args;
    for (auto arg_type : op->arg_types) {
      args.push_back(this->Mutate(arg_type));
    }

    return FuncTypeNode::make(tvm::Array<Type>(args), Mutate(op->ret_type),
                              {}, {});  // fix me
  }

  Type VisitType_(const TupleTypeNode* op, const Type & self) override {
    std::vector<Type> new_fields;
    for (const Type& t : op->fields) {
      new_fields.push_back(this->Mutate(t));
    }
    return TupleTypeNode::make(new_fields);
  }

  Type VisitType_(const TypeRelationNode* op, const Type & self) override {
    return self;
  }

  Type VisitType_(const IncompleteTypeNode* op, const Type & self) override {
    return self;
  }
};

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_PASS_TYPE_VISITOR_H_
