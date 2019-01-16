# pylint: disable=no-else-return, unidiomatic-typecheck, undefined-variable
"""The interface of expr function exposed from C++."""
from tvm._ffi.function import _init_api
from ..relay import ir_pass
from ..relay.backend.interpreter import TensorValue
from ..relay.module import Module
from ..relay.expr import GlobalVar, Function

import numpy as np

_init_api("relay._runtime", __name__)

def optimize(expr, mod=None):
   # TODO: We need to move this optimization code into the optimizer/pass manager
    ck_expr = ir_pass.infer_type(expr, mod=mod)
    fused_expr = ir_pass.fuse_ops(ck_expr)
    ck_fused = ir_pass.infer_type(fused_expr, mod=mod)
    return ck_fused

def test_vm(expr, *args, mod=None):
    cargs = []
    for arg in args:
        if isinstance(arg, np.ndarray):
            cargs.append(TensorValue(arg))
        else:
            assert False

    if mod is None:
        mod = Module.from_expr(expr)

    expr = optimize(expr, mod)
    main = mod.get_global_var('main')

    mod[main] = Function([], expr)
    import pdb; pdb.set_trace()
    return _testeval(mod, cargs)
