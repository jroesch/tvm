import pytest
import tvm

from tvm import IRModule, te, relay, parser
from tvm.relay import op, transform, analysis
from tvm.relay import Any

def infer_mod(mod, annotate_spans=True, use_reference=False):
    if annotate_spans:
        mod = relay.transform.AnnotateSpans()(mod)

    if use_reference:
        mod =
    else:
        mod = transform.InferType()(mod)
    return mod

def infer_expr(expr, annotate_spans=True):
    mod = IRModule.from_expr(expr)
    mod = infer_mod(mod, annotate_spans)
    mod = transform.InferType()(mod)
    entry = mod["main"]
    return entry if isinstance(expr, relay.Function) else entry.body


def assert_has_type(expr, typ, mod=None):
    if not mod:
        mod = tvm.IRModule({})

    mod["main"] = expr
    mod = infer_mod(mod)
    checked_expr = mod["main"]
    checked_type = checked_expr.checked_type
    if checked_type != typ:
        raise RuntimeError("Type mismatch %s vs %s" % (checked_type, typ))

def assert_module_type_checks(program: str) -> None:
    pass

def assert_expr_has_type(expr: str, ty: relay.Type) -> None:
    expr = parser.parse_expr(expr)
    checked_ty = infer_expr(expr, True)
    assert checked_ty == ty, f"Type mismatch `{checked_ty}` vs `{ty}`."
