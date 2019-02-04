# pylint: disable=no-else-return, unidiomatic-typecheck, undefined-variable
"""The interface of expr function exposed from C++."""
from tvm._ffi.function import _init_api
from ..relay import ir_pass
from ..relay.backend.interpreter import TensorValue, TupleValue
from ..relay.module import Module
from ..relay.expr import GlobalVar, Function, var, Call, Expr
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

def _convert(arg, cargs):
    if isinstance(arg, np.ndarray):
        cargs.append(TensorValue(arg))
    elif isinstance(arg, tuple):
        field_args = []
        for field in arg:
            _convert(field, field_args)
        cargs.append(TupleValue(*field_args))
    else:
        raise "unsupported type"

def convert(args):
    cargs = []
    for arg in args:
        _convert(arg, cargs)
    return cargs

def eval_vm(expr_or_mod, ctx, *args):
    if isinstance(expr_or_mod, Expr):
        mod = Module.from_expr(expr_or_mod)
    else:
        mod = expr_or_mod

    main_func = mod[mod.entry_func]

    if len(main_func.params) == 0 and isinstance(main_func.body, GlobalVar):
        main_func = eta_expand(main_func.body, mod)

    assert isinstance(main_func, Function)
    main_func = optimize(mod[mod.entry_func], mod)
    mod[mod.entry_func] = main_func

    cargs = convert(list(args))
#    import pdb; pdb.set_trace()
    return _evaluate_vm(mod, ctx.device_type, ctx.device_id, cargs)
