# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=no-else-return,invalid-name,len-as-condition,too-many-nested-blocks
"""
A pass for manifesting explicit memory allocations.
"""
import attr
import numpy as np
from typing import Optional

from ..expr_functor import ExprMutator
from ..scope_builder import ScopeBuilder
from .. import op, ty, expr
from ... import DataType, register_func, IRModule
from .. import analysis
from . import FoldConstant, InferType, function_pass
from ..backend import compile_engine

def is_primitive(call):
    return hasattr(call, 'op') and hasattr(call.op, 'attrs') and \
           hasattr(call.op.attrs, 'Primitive') and int(call.op.attrs.Primitive) == 1

@attr.s(auto_attribs=True)
class Region:
    var: expr.Var
    size: expr.Expr
    alignment: Optional[expr.Expr]
    dtype: Optional[str]

    def grow(self, size: expr.Expr, alignment: expr.Expr, dtype: str) -> None:
        if self.dtype:
            assert self.dtype == dtype, "must have matching dtypes in a region"
        else:
            self.dtype = dtype

        if self.alignment:
            assert analysis.alpha_equal(self.alignment, alignment), "must have matching alignments in a region"
        else:
            self.alignment = alignment

        self.size = self.size + size

    def next_offset(self) -> None:
        return self.size + expr.const(1, dtype="int64")

    def to_expr(self) -> expr.Expr:
        return op.memory.alloc_storage(self.size, self.alignment, self.dtype)

def iterative_let(let, each_binding, kont):
    bindings = []
    while isinstance(let, expr.Let):
        lhs = let.var
        rhs = let.value
        bindings.append(each_binding(lhs, rhs))
        let = let.body

    return kont(bindings, let)

def mk_let(bindings, body):
    for var, value in reversed(bindings):
        body = expr.Let(var, value, body)
    return body

class StorageCoalesce(ExprMutator):
    def __init__(self):
        super().__init__()
        self.regions = []

    def enter_scope(self):
        zero = expr.const(0, dtype="int64")
        region_var = expr.var(f"region{len(self.regions)}")
        region = Region(region_var, zero, None, None)
        self.regions.append(region)

    def exit_scope(self, body: expr.Expr) -> expr.Expr:
        region = self.regions.pop()
        storage_expr = region.to_expr()
        assert storage_expr, "can not be None"
        return expr.Let(region.var, storage_expr, body)

    def current_region(self) -> Region:
        return self.regions[-1]

    def visit_function(self, function):
        if function.attrs and int(function.attrs.Primitive) == 1:
            return super().visit_function(function)
        else:
            self.enter_scope()
            body = self.visit(function.body)
            body = self.exit_scope(body)
            return expr.Function(
                function.params,
                body,
                function.ret_type,
                function.type_params,
                function.attrs)


    def visit_if(self, ite):
        self.enter_scope()
        true_branch = self.visit(ite.true_branch)
        true_branch = self.exit_scope(true_branch)

        self.enter_scope()
        false_branch = self.visit(ite.false_branch)
        false_branch = self.exit_scope(false_branch)

        return expr.If(ite.cond, true_branch, false_branch)

    def visit_let(self, let):
        def _each_binding(lhs, rhs):
            if isinstance(rhs, expr.Call) and rhs.op == op.op.get("memory.alloc_storage"):
                return self.process_alloc_storage(lhs, rhs)
            elif isinstance(rhs, expr.Call) and rhs.op == op.op.get("memory.alloc_tensor"):
                return self.process_alloc_tensor(lhs, rhs)
            else:
                return lhs, rhs

        return iterative_let(let, _each_binding, mk_let)

    def process_alloc_storage(self, lhs, call):
        size, alignment = call.args
        dtype = call.attrs.dtype
        region = self.current_region()
        region.grow(size, alignment, dtype)
        return lhs, region.var

    def process_alloc_tensor(self, lhs, call):
        region = self.current_region()
        offset = region.next_offset()
        _storage, old_offset, shape = call.args
        assert np.asscalar(old_offset.data.asnumpy()) == 0, "no offsets should yet be allocated"
        return lhs, expr.Call(call.op, [region.var, offset, shape], call.attrs, call.type_args)


class MemoryPlanPass(ExprMutator):
    """A pass for coalescing allocations made by the Relay VM."""
    # def visit_let(self, let):
    #     import pdb; pdb.set_trace()
    # pass restore after rebase


def eval_const(mod, func):
    mod["tmp"] = func
    mod = FoldConstant()(mod)
    return mod["tmp"]

def infer_type(mod, func):
    mod["tmp"] = func
    mod = FoldConstant()(mod)
    return mod["tmp"]


@function_pass(opt_level=0)
class MemoryPlan:
    """An explicit pass wrapper around ManifestAlloc."""
    def __init__(self):
        super().__init__()
        pass

    def transform_function(self, func, mod, _):
        # TODO(@jroesch): Is there a way to do one shot initialization, no need to import every time?
        mod.import_from_std("core.rly")
        sc = StorageCoalesce()
        func = sc.visit(func)
        func = infer_type(mod, func)
        func = eval_const(mod, func)
        ea = MemoryPlanPass()
        func = ea.visit(func)
        return func


register_func("relay.transform.MemoryPlan", MemoryPlan)
