"""Utilities package (backward-compatible re-export surface).

This package replaces the former `fxpmath.utils` module while preserving
existing attribute access patterns.
"""

from .common import array_support, bits_len, min_pow2, int_array
from .repr import (
    twos_complement_repr,
    insert_frac_point,
    binary_repr,
    hex_repr,
    base_repr,
    add_binary_prefix,
    complex_repr,
)
from .parse import (
    strbin2int,
    strbin2float,
    strbin2complex,
    strhex2int,
    strhex2float,
    str2num,
    get_sizes_from_dtype,
)
from .bitwise import (
    ComplexBitwiseOperationWarning,
    binary_invert,
    binary_and,
    binary_or,
    binary_xor,
    is_complex_data,
    bitwise_result_dtype,
    reset_mixed_complex_bitwise_warning_state,
    warn_mixed_complex_bitwise_once,
    twos_complement_componentwise,
    binary_invert_componentwise,
    binary_op_componentwise,
)
from .numeric import clip, clip_vectorized, int_clip, wrap

__all__ = [
    "ComplexBitwiseOperationWarning",
    "array_support",
    "twos_complement_repr",
    "strbin2int",
    "strbin2float",
    "strbin2complex",
    "strhex2int",
    "strhex2float",
    "str2num",
    "insert_frac_point",
    "binary_repr",
    "hex_repr",
    "base_repr",
    "add_binary_prefix",
    "complex_repr",
    "bits_len",
    "min_pow2",
    "binary_invert",
    "binary_and",
    "binary_or",
    "binary_xor",
    "is_complex_data",
    "bitwise_result_dtype",
    "reset_mixed_complex_bitwise_warning_state",
    "warn_mixed_complex_bitwise_once",
    "twos_complement_componentwise",
    "binary_invert_componentwise",
    "binary_op_componentwise",
    "clip",
    "clip_vectorized",
    "int_clip",
    "wrap",
    "get_sizes_from_dtype",
    "int_array",
]
