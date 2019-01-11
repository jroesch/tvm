# pylint: disable=no-else-return, unidiomatic-typecheck, undefined-variable
"""The interface of expr function exposed from C++."""
from tvm._ffi.function import _init_api

_init_api("relay._runtime", __name__)

def test_vm(expr, args):
    return _testeval(expr, args)
