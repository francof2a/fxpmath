"""Public operations facade.

This module re-exports public operator functions from the internal
`fxpmath.operators` package to preserve backward-compatible import paths.
"""

from .objects import Fxp
from .operators import (
    fxp_like,
    fxp_sum,
    from_bin,
    fxp_max,
    fxp_min,
    add,
    sub,
    mul,
    truediv,
    floordiv,
    mod,
    pow,
    sum,
    cumsum,
    cumprod,
    sort,
    conjugate,
    transpose,
    clip,
    diagonal,
    trace,
    prod,
    dot,
    nonzero,
    reshape,
)

__all__ = [
    "Fxp",
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
