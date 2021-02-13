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
"""Test that type checker correctly computes types
   for expressions.
"""
import pytest
import tvm

from tvm import IRModule, te, relay, parser
from tvm.relay import op, transform, analysis
from tvm.relay import Any

from .util import assert_expr_has_type, assert_module_type_checks

@tvm.register_func("tyck.compute_output_shape")
def _compute_output_shape(call):
    func = tvm.get_global_func("ir.OpGetAttr")
    ftvm_compute = func(call.op, "FTVMCompute")
    inputs = []

    for arg in call.args:
        inputs.append(tvm.te.placeholder(arg.type_annotation.shape, arg.type_annotation.dtype))

    outputs = ftvm_compute(call.attrs, inputs, None)
    outs = [(out.shape, out.dtype) for out in outputs]
    return outs

def test_monomorphic_let():
    assert_expr_has_type(
        "let %x: float32 = 1f; %x",
        "float32")


def test_single_op():
    assert_expr_has_type(
        "fn (%x : float32) { let %t1 = log(%x); %t1 }",
        "fn (float32) -> float32")


def test_fn_decl():
    assert_expr_has_type(
     """fn @f(%x : Tensor[(10, 10), float32]) {
         log(%x)
     }
     """,
     """fn (%x: Tensor[(10, 10), float32]) -> Tensor[(10, 10), float32]""")

def test_broadcast_op():
    assert_expr_has_type(
        """
        fn (%x: Tensor[(10, 4), float32], %y: Tensor[(5, 10, 1), float32])
            -> Tensor[(5, 10, 4), float32] {
            %x + %y
        }
        """,
        "fn (Tensor[(10, 4), float32], Tensor[(5, 10, 1), float32]) -> Tensor[(5, 10, 4), float32]")

def let_bind_elemwise_calls():
    assert_expr_has_type(
        """
        fn (%x : Tensor[(10, 10), float32]) {
           let %t1 = log(x);
           let %t2 = add(%t1, %x);
           %t1
        }""",
        "fn (Tensor[(10, 10), float32] -> Tensor[(10, 10), float32]")

def test_tuple():
    assert_expr_has_type(
        """
        free_var %x: Tensor[(10,), float32]
        (%x, %x)
        """,
        """(Tensor[(10,), float32], Tensor[(10,), float32])"""
    )

def test_ref():
    assert_expr_has_type(
        """
        free_var %x: float32;
        free_var %y: float32;
        let %r = ref(%x);
        let %value = ref_read(%r);
        let %unit = ref_write(%r, %y);
        (%value, %unit)
        """,
        """(float32, ())""")

# def test_free_expr():
#     x = relay.var("x", "float32")
#     y = relay.add(x, x)
#     yy = infer_expr(y, annotate_spans=False)
#     assert tvm.ir.structural_equal(yy.args[0], x, map_free_vars=True)
#     assert yy.checked_type == relay.scalar_type("float32")
#     assert x.vid.same_as(yy.args[0].vid)


# def test_type_args():
#     x = relay.var("x", shape=(10, 10))
#     y = relay.var("y", shape=(1, 10))
#     z = relay.add(x, y)
#     ty_z = infer_expr(z)
#     ty_args = ty_z.type_args
#     assert len(ty_args) == 2
#     assert ty_args[0].dtype == "float32"
#     assert ty_args[1].dtype == "float32"
#     sh1 = ty_args[0].shape
#     sh2 = ty_args[1].shape
#     assert sh1[0].value == 10
#     assert sh1[1].value == 10
#     assert sh2[0].value == 1
#     assert sh2[1].value == 10


def test_recursion():
    assert_module_type_checks(
        """
        #[version = "0.0.5"]
        def @f(%n: int32, %data: float32) -> float32 {
            if (%n == 0) {
                %data
            } else {
                @f(%n - 1, log(%data))
            }
        }
        """)

def test_incomplete_call():
    # TODO(@jroesch): is this the right test?
    assert_module_type_checks(
        """
        #[version = "0.0.5"]
        def @f(%x: int32) {
            @f(%x)
        }
        """)


def test_equal():
    i = relay.var("i", shape=[], dtype="int32")
    eq = op.equal(i, relay.const(0, dtype="int32"))
    func = relay.Function([i], eq)
    ft = infer_expr(func)
    expected = relay.FuncType([relay.scalar_type("int32")], relay.scalar_type("bool"))
    assert ft.checked_type == expected

    assert ft.checked_type == relay.FuncType(
        [relay.scalar_type("int32")], relay.scalar_type("bool")
    )

# def test_higher_order_argument():
#     a = relay.TypeVar("a")
#     x = relay.Var("x", a)
#     id_func = relay.Function([x], x, a, [a])

#     b = relay.TypeVar("b")
#     f = relay.Var("f", relay.FuncType([b], b))
#     y = relay.Var("y", b)
#     ho_func = relay.Function([f, y], f(y), b, [b])

#     # id func should be an acceptable argument to the higher-order
#     # function even though id_func takes a type parameter
#     ho_call = ho_func(id_func, relay.const(0, "int32"))

#     hc = infer_expr(ho_call)
#     expected = relay.scalar_type("int32")
#     assert hc.checked_type == expected


# def test_higher_order_return():
#     a = relay.TypeVar("a")
#     x = relay.Var("x", a)
#     id_func = relay.Function([x], x, a, [a])

#     b = relay.TypeVar("b")
#     nested_id = relay.Function([], id_func, relay.FuncType([b], b), [b])

#     ft = infer_expr(nested_id)
#     assert ft.checked_type == relay.FuncType([], relay.FuncType([b], b), [b])


# def test_higher_order_nested():
#     a = relay.TypeVar("a")
#     x = relay.Var("x", a)
#     id_func = relay.Function([x], x, a, [a])

#     choice_t = relay.FuncType([], relay.scalar_type("bool"))
#     f = relay.Var("f", choice_t)

#     b = relay.TypeVar("b")
#     z = relay.Var("z")
#     top = relay.Function(
#         [f], relay.If(f(), id_func, relay.Function([z], z)), relay.FuncType([b], b), [b]
#     )

#     expected = relay.FuncType([choice_t], relay.FuncType([b], b), [b])
#     ft = infer_expr(top)
#     assert ft.checked_type == expected




# def test_global_var_recursion():
#     mod = tvm.IRModule({})
#     gv = relay.GlobalVar("main")
#     x = relay.var("x", shape=[])
#     tt = relay.scalar_type("float32")

#     func = relay.Function([x], relay.Call(gv, [x]), tt)
#     mod[gv] = func
#     mod = infer_mod(mod)
#     func_ty = mod["main"].checked_type

#     assert func_ty == relay.FuncType([tt], tt)



# def initialize_box_adt(mod):
#     # initializes simple ADT for tests
#     box = relay.GlobalTypeVar("box")
#     tv = relay.TypeVar("tv")
#     constructor = relay.Constructor("constructor", [tv], box)
#     data = relay.TypeData(box, [tv], [constructor])
#     mod[box] = data
#     return box, constructor

# def test_constructor_type():
#     mod = tvm.IRModule()
#     box, constructor = initialize_box_adt(mod)

#     a = relay.TypeVar("a")
#     x = relay.Var("x", a)
#     func = relay.Function([x], constructor(x), box(a), [a])
#     mod["main"] = func
#     mod = infer_mod(mod)
#     func_ty = mod["main"].checked_type
#     box = mod.get_global_type_var("box")
#     expected = relay.FuncType([a], box(a), [a])
#     assert func_ty == expected


# def test_constructor_call():
#     mod = tvm.IRModule()
#     box, constructor = initialize_box_adt(mod)

#     box_unit = constructor(relay.Tuple([]))
#     box_constant = constructor(relay.const(0, "float32"))

#     func = relay.Function([], relay.Tuple([box_unit, box_constant]))
#     mod["main"] = func
#     mod = infer_mod(mod)
#     ret_type = mod["main"].checked_type.ret_type.fields
#     # NB(@jroesch): when we annotate spans the ast fragments before
#     # annotation the previous fragments will no longer be directly equal.
#     box = mod.get_global_type_var("box")
#     expected1 = box(relay.TupleType([]))
#     expected2 = box(relay.TensorType((), "float32"))
#     assert ret_type[0] == expected1
#     assert ret_type[1] == expected2


# def test_adt_match():
#     mod = tvm.IRModule()
#     box, constructor = initialize_box_adt(mod)

#     v = relay.Var("v", relay.TensorType((), "float32"))
#     match = relay.Match(
#         constructor(relay.const(0, "float32")),
#         [
#             relay.Clause(
#                 relay.PatternConstructor(constructor, [relay.PatternVar(v)]), relay.Tuple([])
#             ),
#             # redundant but shouldn't matter to typechecking
#             relay.Clause(relay.PatternWildcard(), relay.Tuple([])),
#         ],
#     )

#     func = relay.Function([], match)
#     mod["main"] = func
#     mod = infer_mod(mod)
#     actual = mod["main"].checked_type.ret_type
#     assert actual == relay.TupleType([])


# def test_adt_match_type_annotations():
#     mod = tvm.IRModule()
#     box, constructor = initialize_box_adt(mod)

#     # the only type annotation is inside the match pattern var
#     # but that should be enough info
#     tt = relay.TensorType((2, 2), "float32")
#     x = relay.Var("x")
#     mv = relay.Var("mv", tt)
#     match = relay.Match(
#         constructor(x),
#         [
#             relay.Clause(
#                 relay.PatternConstructor(constructor, [relay.PatternVar(mv)]), relay.Tuple([])
#             )
#         ],
#     )

#     mod["main"] = relay.Function([x], match)
#     mod = infer_mod(mod)
#     ft = mod["main"].checked_type
#     assert ft == relay.FuncType([tt], relay.TupleType([]))


# def test_let_polymorphism():
#     id = relay.Var("id")
#     xt = relay.TypeVar("xt")
#     x = relay.Var("x", xt)
#     body = relay.Tuple([id(relay.const(1)), id(relay.Tuple([]))])
#     body = relay.Let(id, relay.Function([x], x, xt, [xt]), body)
#     body = infer_expr(body)
#     int32 = relay.TensorType((), "int32")
#     tvm.ir.assert_structural_equal(body.checked_type, relay.TupleType([int32, relay.TupleType([])]))


# def test_if():
#     choice_t = relay.FuncType([], relay.scalar_type("bool"))
#     f = relay.Var("f", choice_t)
#     true_branch = relay.Var("True", relay.TensorType([Any(), 1], dtype="float32"))
#     false_branch = relay.Var("False", relay.TensorType([Any(), Any()], dtype="float32"))
#     top = relay.Function([f, true_branch, false_branch], relay.If(f(), true_branch, false_branch))
#     ft = infer_expr(top)
#     tvm.ir.assert_structural_equal(ft.ret_type, relay.TensorType([Any(), 1], dtype="float32"))


# TODO(@jroesch): this one fails due to parsing issue
def test_type_arg_infer():
    mod = assert_module_type_checks(
        """
        #[version = "0.0.5"]
        def @id[A](%x: A) -> A {
            %x
        }

        def @main(%f: float32) -> float32 {
            @id(%f)
        }
        """)
    tvm.ir.assert_structural_equal(mod["main"].body.type_args, [relay.TensorType((), "float32")])

if __name__ == "__main__":
    import sys
    pytest.main(sys.argv)
