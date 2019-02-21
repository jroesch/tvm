# pylint: disable=no-else-return, unidiomatic-typecheck, undefined-variable
"""The interface of expr function exposed from C++."""
import tvm
from tvm._ffi.function import _init_api, _VMObjectBase, _set_vm_obj_function
from ..relay import ir_pass
from ..relay.backend.interpreter import TensorValue, TupleValue, Executor
from ..relay.module import Module
from ..relay.expr import GlobalVar, Function, var, Call, Expr
from ..relay.ty import FuncType
from . import _vm

import numpy as np

class VMObject(_VMObjectBase):
    def to_value(self):
        return _vm._VMToValue(self)

    def tag(self):
        return _vm._VMObjectTag(self)

_set_vm_obj_function(VMObject)

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
        cargs.append(_vm._Tensor(tvm.nd.array(arg)))
    elif isinstance(arg, tuple):
        field_args = []
        for field in arg:
            _convert(field, field_args)
        cargs.append(_vm._Tuple(*field_args))
    else:
        raise "unsupported type"

def convert(args):
    cargs = []
    for arg in args:
        _convert(arg, cargs)
    return cargs

def eval_vm(mod, ctx, *args):
    """
    Evaluate a module on a given context with the provided arguments.

    Parameters
    ----------
    mod: relay.Module
        The module to optimize, will execute its entry_func.
    ctx: tvm.Context
        The TVM context to execute on.
    args: ...
        The arguments to evaluate.
    """
    main_func = mod[mod.entry_func]

    if len(main_func.params) == 0 and isinstance(main_func.body, GlobalVar):
        main_func = eta_expand(main_func.body, mod)

    assert isinstance(main_func, Function)
    main_func = optimize(mod[mod.entry_func], mod)
    mod[mod.entry_func] = main_func

    cargs = convert(list(args))
    import pdb; pdb.set_trace()

    result = _vm._evaluate_vm(mod, ctx.device_type, ctx.device_id, *cargs)
    return result

class VMExecutor(Executor):
    """
    An implementation of the executor interface for
    the Relay VM.

    Useful interface for experimentation and debugging
    the VM can also be used directly from the API.
    supported by `tvm.relay.vm`.
    """

    """
    Parameters
    ----------
    mod : :py:class:`~tvm.relay.module.Module`
        The module to support the execution.

    ctx : :py:class:`TVMContext`
        The runtime context to run the code on.

    target : :py:class:`Target`
        The target option to build the function.
    """
    def __init__(self, mod, ctx, target):
        self.mod = mod
        self.ctx = ctx
        self.target = target

    def _make_executor(self, expr):
        assert isinstance(expr, Expr)
        self.mod[self.mod.entry_func] = expr
        main = self.mod[self.mod.entry_func]

        def _vm_wrapper(*args, **kwargs):
            args = self._convert_args(main, args, kwargs)
            return eval_vm(self.mod, self.ctx, args)

        return _vm_wrapper

