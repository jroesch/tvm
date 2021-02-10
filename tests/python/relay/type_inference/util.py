import pytest
import tvm

from tvm import IRModule, te, relay, parser
from tvm.contrib import rust
from tvm.relay import op, transform, analysis
from tvm.relay import Any

def infer_mod(mod, annotate_spans=True, use_reference=False):
    if use_reference:
        print("Using Reference Type Checker ...")
        ty_infer = rust.InferType()
    else:
        ty_infer = transform.InferType()


    if annotate_spans:
        mod = relay.transform.AnnotateSpans()(mod)

    return ty_infer(mod)

def infer_expr(expr, annotate_spans=True, use_reference=False):
    mod = IRModule.from_expr(expr)
    mod = infer_mod(mod, annotate_spans, use_reference)
    entry = mod["main"]
    return entry if isinstance(expr, relay.Function) else entry.body


def assert_module_type_checks(program: str) -> IRModule:
    mod = parser.parse(program)
    return infer_mod(mod, annotate_spans=True, use_reference=True)


def assert_expr_has_type(expr: str, ty: str) -> None:
    ty = parser.parse_type(ty)
    expr = parser.parse_expr(expr)
    checked_expr = infer_expr(expr, annotate_spans=True, use_reference=True)
    checked_type = checked_expr.checked_type
    print(f"Checked Type: {checked_type}")
    print(f"Expected Type: {ty}")
    assert checked_type == ty, f"Type mismatch `{checked_type}` vs `{ty}`."
