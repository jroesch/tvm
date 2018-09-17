/*!
 *  Copyright (c) 2018 by Contributors
 * \file tvm/relay/expr_visitor.h
 * \brief A simple visitor wrapper around ExprFunctor.
 *
 * Exposes two visitors with default traversal strategies, one
 * which doesn't compute a result but can mutate internal state,
 * and another which functionally builds a new Expr.
 */
#ifndef TVM_RELAY_EXPR_VISITOR_H_
#define TVM_RELAY_EXPR_VISITOR_H_

#include <tvm/relay/expr_functor.h>

namespace tvm {
namespace relay {

class ExprVisitor : public ::tvm::relay::ExprFunctor<void(const Expr& n)> {
 public:
  void VisitExpr_(const VarNode* op) override { return; }

  void VisitExpr_(const GlobalVarNode* op) override { return; }

  void VisitExpr_(const ConstantNode* op) override { return; }

  void VisitExpr_(const TupleNode* op) override {
    for (auto field : op->fields) {
      this->VisitExpr(field);
    }
  }

  void VisitExpr_(const ParamNode* op) override { this->VisitExpr(op->var); }

  void VisitExpr_(const FunctionNode* op) override {
    for (auto param : op->params) {
      this->VisitExpr(param);
    }

    this->VisitExpr(op->body);
  }

  void VisitExpr_(const CallNode* op) override {
    this->VisitExpr(op->op);
    for (auto ty_arg : op->type_args) {
      this->VisitType(ty_arg);
    }

    for (auto arg : op->args) {
      this->VisitExpr(arg);
    }
  }

  void VisitExpr_(const LetNode* op) override {
    this->VisitExpr(op->var);
    this->VisitExpr(op->value);
    this->VisitExpr(op->body);
  }

  void VisitExpr_(const IfNode* op) override {
    this->VisitExpr(op->cond);
    this->VisitExpr(op->true_branch);
    this->VisitExpr(op->false_branch);
  }

  void VisitExpr_(const OpNode* op) override { return; }

  virtual void VisitType(const Type& t) {}
};

struct ShallowHashConser : ExprFunctor<Expr(const Expr & ret, const Expr & self, const Expr & orig)> {
  Expr VisitExpr_(const VarNode * ret, const Expr & self, const Expr & orig) override {
    return self;
  }

  Expr VisitExpr_(const ConstantNode * ret, const Expr & self, const Expr & orig) override {
    if (auto p = self.as<ConstantNode>()) {
      
    }
    // todo: how?
    return self;
  }

  Expr VisitExpr_(const GlobalVarNode * op, const Expr & self, const Expr & orig) override {
    return self;
  }

  Expr VisitExpr_(const OpNode * op, const Expr & self, const Expr & orig) override {
    return self;
  }

  Expr VisitExpr_(const TupleNode * op, const Expr & self, const Expr & orig) override {
    if (auto p = orig.as<TupleNode>()) {
      if (op->fields.size() != p->fields.size()) {
        return self;
      }
      for (size_t i = op->fields.size(); i < op->fields.size(); ++i) {
        if (op->fields[i] != p->fields[i]) {
          return self;
        }
      }
      return orig;
    } else {
      return self;
    }
  }

  Expr VisitExpr_(const ParamNode* op, const Expr & self, const Expr & orig) override {
    if (auto p = orig.as<ParamNode>()) {
      if (op->var == p->var && op->type == p->type) {
        return orig;
      }
    }
    return self;
  }

  Expr VisitExpr_(const FunctionNode* op, const Expr & self, const Expr & orig) override {
    if (auto p = orig.as<FunctionNode>()) {
      if (op->type_params.size() != p->type_params.size()) {
        return self;
      }
      for (size_t i = 0; i < op->type_params.size(); ++i) {
        if (op->type_params[i] != p->type_params[i]) {
          return self;
        }
      }
      if (op->params.size() != p->params.size()) {
        return self;
      }
      for (size_t i = 0; i < op->params.size(); ++i) {
        if (op->params[i] != p->params[i]) {
          return self;
        }
      }
      if (op->ret_type == p->ret_type && op->body == p->body) {
        return orig;
      }
    }
    return self;
  }

  Expr VisitExpr_(const CallNode* op, const Expr & self, const Expr & orig) override {
    if (auto p = orig.as<CallNode>()) {
      if (op->op != p->op) {
        return self;
      }
      if (op->type_args.size() != p->type_args.size()) {
        return self;
      }
      for (size_t i = 0; i < op->type_args.size(); ++i) {
        if (op->type_args[i] != p->type_args[i]) {
          return self;
        }
      }
      if (op->args.size() != p->args.size()) {
        return self;
      }
      for (size_t i = 0; i < op->args.size(); ++i) {
        if (op->args[i] != p->args[i]) {
          return self;
        }
      }
      if (op->attrs == p->attrs) {
        return orig;
      }
    }
    return self;
  }

  Expr VisitExpr_(const LetNode* op, const Expr & self, const Expr & orig) override {
    if (auto p = orig.as<LetNode>()) {
      if (op->var == p->var && op->value_type == p->value_type && op->value == p->value && op->body == p->body) {
        return orig;
      }
    }
    return self;
  }

  Expr VisitExpr_(const IfNode* op, const Expr & self, const Expr & orig) override {
    if (auto p = orig.as<IfNode>()) {
      if (op->cond == p->cond && op->true_branch == p->true_branch && op->false_branch == p->false_branch) {
        return orig;
      }
    }
    return self;
  }

};

inline Expr ShallowHashCons(const Expr & ret, const Expr & orig) {
  return ShallowHashConser()(ret, ret, orig);
}

// Like IRMutator, ExprMutator return the old expr if result is structurally equivalent.
// However, since we treat Expr as a tree, it will still be type checked/evaluated multiple time.
// Use Let if you want to express sharing.
class ExprMutator : public ::tvm::relay::ExprFunctor<Expr(const Expr& n, const Expr & self)> {
 public:
  Expr Mutate(const Expr & self) {
    return ShallowHashCons(this->VisitExpr(self, self), self);
  }

  Expr VisitExpr_(const VarNode* op, const Expr & self) override {
    return self;
  }

  Expr VisitExpr_(const ConstantNode* op, const Expr & self) override {
    return self;
  }

  Expr VisitExpr_(const GlobalVarNode* op, const Expr & self) override {
    return self;
  }

  Expr VisitExpr_(const OpNode* op, const Expr & self) override {
    return self;
  }

  Expr VisitExpr_(const TupleNode* op, const Expr & self) override {
    tvm::Array<Expr> fields;
    for (auto field : op->fields) {
      fields.push_back(this->Mutate(field));
    }

    return TupleNode::make(fields);
  }

  Expr VisitExpr_(const ParamNode* op, const Expr & self) override {
    Expr var_expr = this->Mutate(op->var);
    if (const VarNode* var_node = var_expr.as<VarNode>()) {
      auto var = GetRef<Var>(var_node);
      auto type = this->VisitType(op->type);
      return ParamNode::make(var, type);
    } else {
      LOG(FATAL) << "the default param visitor expected a Var found: "
                 << var_expr << std::endl;
      return Expr();
    }
  }

  Expr VisitExpr_(const FunctionNode* op, const Expr & self) override {
    tvm::Array<TypeParam> ty_params;

    for (auto ty : op->type_params) {
      Type ty_param_type = VisitType(ty);
      if (auto ty_param = ty_param_type.as<TypeParamNode>()) {
        auto ty_param_ref = GetRef<TypeParam>(ty_param);
        ty_params.push_back(ty_param_ref);
      } else {
        LOG(FATAL)
            << "the default function visitor expected a TypeParam found: "
            << ty_param_type << std::endl;
        return Expr();
      }
    }

    tvm::Array<Param> params;
    for (auto param : op->params) {
      Expr param_expr = this->Mutate(param);
      if (const ParamNode* param_node = param_expr.as<ParamNode>()) {
        auto param = GetRef<Param>(param_node);
        params.push_back(param);
      } else {
        CHECK(false) << "the default function visitor expected a Param found: "
                     << param_expr << std::endl;
        return Expr();
      }
    }

    auto ret_type = this->VisitType(op->ret_type);
    auto body = this->Mutate(op->body);
    return FunctionNode::make(params, ret_type, body, ty_params);
  }

  Expr VisitExpr_(const CallNode* call_node, const Expr & self) override {
    auto fn = this->Mutate(call_node->op);

    tvm::Array<Type> ty_args;
    for (auto ty_arg : call_node->type_args) {
      auto new_ty_arg = this->VisitType(ty_arg);
      ty_args.push_back(new_ty_arg);
    }

    tvm::Array<Expr> call_args;
    for (auto arg : call_node->args) {
      call_args.push_back(this->Mutate(arg));
    }

    auto call = CallNode::make(fn, call_args, call_node->attrs, ty_args);

    return call;
  }

  Expr VisitExpr_(const LetNode* op, const Expr & self) override {
    Expr var_expr = this->Mutate(op->var);
    if (const VarNode* var_node = var_expr.as<VarNode>()) {
      auto var = GetRef<Var>(var_node);
      auto type = this->VisitType(op->value_type);
      auto value = this->Mutate(op->value);
      auto body = this->Mutate(op->body);
      return LetNode::make(var, value, body, type);
    } else {
      LOG(FATAL) << "the default let visitor expected a Var found: "
                   << var_expr << std::endl;
      return Expr();
    }
  }

  Expr VisitExpr_(const IfNode* op, const Expr & self) override {
    auto guard = this->Mutate(op->cond);
    auto true_b = this->Mutate(op->true_branch);
    auto false_b = this->Mutate(op->false_branch);
    return IfNode::make(guard, true_b, false_b);
  }

  virtual Type VisitType(const Type& t) { return t; }
};

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_EXPR_VISITOR_H_
