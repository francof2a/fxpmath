"""Bitwise utility helpers, including complex/component-wise behavior."""

import numpy as np
import warnings

from .common import bits_len
from .repr import twos_complement_repr

_mixed_complex_bitwise_warned = False
class ComplexBitwiseOperationWarning(UserWarning):
    """Warning emitted once when mixing complex and non-complex bitwise operands."""


def _bitwise_infer_n_word(*vals):
    """Infer bit width for bitwise ops when n_word is omitted."""
    inferred = 0
    for val in vals:
        arr = np.asarray(val)
        if arr.ndim == 0:
            inferred = max(inferred, bits_len(arr.item()))
        elif arr.size > 0:
            for item in arr.flat:
                inferred = max(inferred, bits_len(item))
    return inferred


def _bitwise_binary_apply(x, y, n_word, pyop):
    """Apply a binary bitwise op with NumPy broadcasting semantics."""
    if n_word is None:
        n_word = _bitwise_infer_n_word(x, y)
    n_word = int(n_word)
    mod = 1 << n_word

    xa = np.asarray(x)
    ya = np.asarray(y)
    x_b, y_b = np.broadcast_arrays(xa, ya)

    op = np.frompyfunc(lambda a, b: pyop(int(a) % mod, int(b) % mod), 2, 1)
    z = np.asarray(op(x_b, y_b), dtype=object)

    if xa.ndim == 0 and ya.ndim == 0:
        return int(z.item())
    return z


def _bitwise_unary_apply(x, n_word, pyop):
    """Apply a unary bitwise op over scalar or array inputs."""
    if n_word is None:
        n_word = _bitwise_infer_n_word(x)
    n_word = int(n_word)
    mod = 1 << n_word

    xa = np.asarray(x)
    op = np.frompyfunc(lambda a: pyop(int(a) % mod), 1, 1)
    z = np.asarray(op(xa), dtype=object)

    if xa.ndim == 0:
        return int(z.item())
    return z


def binary_invert(x, n_word=None):
    """Apply bitwise NOT with broadcasting-friendly behavior."""
    if n_word is None:
        n_word = _bitwise_infer_n_word(x)
    n_word = int(n_word)
    mod_minus_one = (1 << n_word) - 1
    return _bitwise_unary_apply(x, n_word=n_word, pyop=lambda a: mod_minus_one - a)


def binary_and(x, y, n_word=None):
    """Apply bitwise AND with NumPy broadcasting semantics."""
    return _bitwise_binary_apply(x, y, n_word=n_word, pyop=lambda a, b: a & b)


def binary_or(x, y, n_word=None):
    """Apply bitwise OR with NumPy broadcasting semantics."""
    return _bitwise_binary_apply(x, y, n_word=n_word, pyop=lambda a, b: a | b)


def binary_xor(x, y, n_word=None):
    """Apply bitwise XOR with NumPy broadcasting semantics."""
    return _bitwise_binary_apply(x, y, n_word=n_word, pyop=lambda a, b: a ^ b)

def is_complex_data(x):
    """Return True when an operand contains complex values."""
    return isinstance(x, complex) or np.iscomplexobj(x)


def bitwise_result_dtype(base_vdtype, force_complex=False):
    """Preserve base vdtype unless operation requires complex output."""
    if force_complex:
        return complex
    return base_vdtype


def reset_mixed_complex_bitwise_warning_state():
    """Reset the one-time mixed-complex bitwise warning flag."""
    global _mixed_complex_bitwise_warned
    _mixed_complex_bitwise_warned = False


def warn_mixed_complex_bitwise_once(stacklevel=2):
    """Warn once when bitwise ops mix complex and non-complex operands."""
    global _mixed_complex_bitwise_warned
    if not _mixed_complex_bitwise_warned:
        warnings.warn(
            'Bitwise operation mixed complex and non-complex operands; applying the real operand to both real and imaginary parts.',
            ComplexBitwiseOperationWarning,
            stacklevel=stacklevel,
        )
        _mixed_complex_bitwise_warned = True


def twos_complement_componentwise(val, nbits):
    """Apply two's-complement conversion to real/imag parts independently."""
    if is_complex_data(val):
        real_val = twos_complement_repr(np.real(val), nbits=nbits)
        imag_val = twos_complement_repr(np.imag(val), nbits=nbits)
        return real_val + 1j * imag_val
    return twos_complement_repr(val, nbits=nbits)


def binary_invert_componentwise(x, n_word=None):
    """Apply bitwise invert, including component-wise handling for complex values."""
    if is_complex_data(x):
        real_val = binary_invert(np.real(x), n_word=n_word)
        imag_val = binary_invert(np.imag(x), n_word=n_word)
        return real_val + 1j * imag_val, True

    return binary_invert(x, n_word=n_word), False


def binary_op_componentwise(x, y, op, n_word=None, warn_mixed=True, warning_stacklevel=2):
    """Apply a binary bitwise op with complex component-wise semantics."""
    x_is_complex = is_complex_data(x)
    y_is_complex = is_complex_data(y)
    force_complex = x_is_complex or y_is_complex

    if not force_complex:
        return op(x, y, n_word=n_word), False

    if x_is_complex and y_is_complex:
        left_real = np.real(x)
        left_imag = np.imag(x)
        right_real = np.real(y)
        right_imag = np.imag(y)
    elif x_is_complex:
        if warn_mixed:
            warn_mixed_complex_bitwise_once(stacklevel=warning_stacklevel)
        left_real = np.real(x)
        left_imag = np.imag(x)
        right_real = y
        right_imag = y
    else:
        if warn_mixed:
            warn_mixed_complex_bitwise_once(stacklevel=warning_stacklevel)
        left_real = x
        left_imag = x
        right_real = np.real(y)
        right_imag = np.imag(y)

    out_real = op(left_real, right_real, n_word=n_word)
    out_imag = op(left_imag, right_imag, n_word=n_word)
    return out_real + 1j * out_imag, True

