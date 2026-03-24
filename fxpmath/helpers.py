"""Internal helpers for fixed-point operation kernels."""

import numpy as np

from . import _n_word_max


def _cast_to_object(x):
    """Cast inputs to NumPy object dtype to avoid overflow in wide integer operations.
    
    Parameters
    ---
    x : array_like
        Scalar or array to convert into an object-dtype NumPy array.
    
    Returns
    ---
    numpy.ndarray
        Array view/copy of `x` with `dtype=object`."""
    # Normalize NumPy scalar objects (e.g., np.int64) to Python scalars so
    # object-mode arithmetic uses arbitrary-precision Python ints/floats.
    if isinstance(x, np.generic):
        x = x.item()
    return np.asarray(x).astype(object)


def _cast_func(use_object):
    """Return a casting helper that optionally converts arrays to object dtype.
    
    Parameters
    ---
    use_object : bool
        When `True`, the returned callable casts inputs with `_cast_to_object`.
        When `False`, the returned callable returns inputs unchanged.
    
    Returns
    ---
    Callable[[array_like], array_like]
        Casting helper used by arithmetic kernels."""
    return _cast_to_object if use_object else (lambda m: m)


def _requires_object_for_scale(n_word, shift):
    """Decide whether a scaling/shift operation needs object dtype to remain safe.
    
    Parameters
    ---
    n_word : int or None
        Operand word length in bits. `None` disables this check and returns `False`.
    shift : int
        Power-of-two scaling shift applied to the operand.
    
    Returns
    ---
    bool
        `True` when the scale operation could exceed native integer safety and
        should use object arithmetic."""
    shift = int(shift)

    if n_word is None:
        return False

    n_word = int(n_word)

    if n_word >= _n_word_max:
        return True

    if shift >= (_n_word_max - 1):
        return True

    if shift > 0 and (n_word + shift) >= _n_word_max:
        return True

    return False


def _use_object_cast(scale_terms=None, product_terms=None, pow2_terms=None):
    """Determine whether any part of an operation requires object-dtype arithmetic.
    
    Parameters
    ---
    scale_terms : sequence[tuple[int | None, int]], optional
        Pairs of `(n_word, shift)` checked with `_requires_object_for_scale`.
    product_terms : sequence[tuple[int, ...] | int], optional
        Terms whose total bit growth is evaluated against `_n_word_max`.
    pow2_terms : sequence[int], optional
        Shifts used in `2**shift` factors; very large shifts force object mode.
    
    Returns
    ---
    bool
        `True` when at least one condition requires object-dtype arithmetic."""
    if scale_terms is not None:
        for n_word, shift in scale_terms:
            if _requires_object_for_scale(n_word, shift):
                return True

    if product_terms is not None:
        for terms in product_terms:
            if isinstance(terms, (tuple, list)):
                total_bits = int(np.sum(terms))
            else:
                total_bits = int(terms)

            if total_bits >= _n_word_max:
                return True

    if pow2_terms is not None:
        for shift in pow2_terms:
            if int(shift) >= (_n_word_max - 1):
                return True

    return False
