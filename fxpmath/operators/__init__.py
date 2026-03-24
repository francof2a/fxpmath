"""Operator submodules and re-exported public operation functions."""

from .core import fxp_like, from_bin
from .reduction import fxp_sum, fxp_max, fxp_min, sum, cumsum, cumprod, prod
from .arithmetic import add, sub, mul, truediv, floordiv, mod, pow
from .array import sort, conjugate, transpose, clip, diagonal, trace, dot, nonzero, reshape

__all__ = [
    "fxp_like",
    "fxp_sum",
    "from_bin",
    "fxp_max",
    "fxp_min",
    "add",
    "sub",
    "mul",
    "truediv",
    "floordiv",
    "mod",
    "pow",
    "sum",
    "cumsum",
    "cumprod",
    "sort",
    "conjugate",
    "transpose",
    "clip",
    "diagonal",
    "trace",
    "prod",
    "dot",
    "nonzero",
    "reshape",
]
