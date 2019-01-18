# pylint: disable=no-else-return, unidiomatic-typecheck, undefined-variable
"""The interface of expr function exposed from C++."""
from tvm._ffi.function import _init_api
from ..relay import ir_pass
from ..relay.backend.interpreter import TensorValue
from ..relay.module import Module
from ..relay.expr import GlobalVar, Function, var, Call
from ..relay.ty import FuncType

import numpy as np

_init_api("relay._vm", __name__)

def optimize(expr, mod=None):
   # TODO: We need to move this optimization code into the optimizer/pass manager
    ck_expr = ir_pass.infer_type(expr, mod=mod)
    fused_expr = ir_pass.fuse_ops(ck_expr, mod=mod)
    ck_fused = ir_pass.infer_type(fused_expr, mod=mod)
    return ck_fused

def eta_expand(expr, mod):
    if isinstance(expr, GlobalVar):
        ck_type = mod[expr].checked_type
    else:
        expr = ir_pass.infer_type(expr, mod)
        ck_type = expr.checked_type

    assert isinstance(ck_type, FuncType)
    eta_args = []
    for arg_type in ck_type.arg_types:
        eta_args.append(var('a', type_annotation=arg_type))

    return Function(eta_args, Call(expr, eta_args))


def eval_vm(expr, *args, mod=None):
    if isinstance(expr, GlobalVar):
        expr = eta_expand(expr, mod)

    assert isinstance(expr, Function)

    cargs = []
    for arg in args:
        if isinstance(arg, np.ndarray):
            cargs.append(TensorValue(arg))
        else:
            assert False


    if mod is None:
        mod = Module.from_expr(expr)

    main = mod.get_global_var('main')
    expr = optimize(expr, mod)

    mod[main] = expr

    return _evaluate_vm(mod, cargs)
