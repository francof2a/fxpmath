"""Numeric clipping/wrapping helpers for fixed-point internals."""

import numpy as np

from .. import _n_word_max
from .common import int_array
@np.vectorize
def clip(x, val_min, val_max):
    """Clip fixed-point values to minimum and maximum bounds.
    
    Parameters
    ---
    x : scalar or numpy.ndarray
        Input value(s) to clip.
    val_min : scalar
        Lower clipping bound.
    val_max : scalar
        Upper clipping bound.
    
    Returns
    ---
    scalar or numpy.ndarray
        Clipped value(s) within [`val_min`, `val_max`]."""
    x_clipped = np.array(max(val_min, min(val_max, x)))
    return x_clipped

@np.vectorize
def int_clip(x, val_min, val_max):
    """Clip integer values between minimum and maximum limits.
    
    Parameters
    ---
    x : int or numpy.ndarray
        Integer value(s) to clip.
    val_min : int
        Lower clipping bound.
    val_max : int
        Upper clipping bound.
    
    Returns
    ---
    int or numpy.ndarray
        Integer-clipped value(s) within bounds."""
    x_clipped = np.array(max(val_min, min(val_max, int(x))))
    return x_clipped

def wrap(x, signed, n_word):

    """Wrap integers into the representable range using modular arithmetic.
    
    Parameters
    ---
    x : int or numpy.ndarray
        Raw integer value(s) to wrap to fixed-point range.
    signed : bool
        Whether to wrap to signed or unsigned range.
    n_word : int
        Word length that defines wrapping period.
    
    Returns
    ---
    int or numpy.ndarray
        Wrapped integer value(s) constrained to fixed-point range."""
    if np.iscomplexobj(x):
        # wrap real and imaginary parts independently
        wrapped_real = wrap(np.real(x), signed=signed, n_word=n_word)
        wrapped_imag = wrap(np.imag(x), signed=signed, n_word=n_word)
        return wrapped_real + 1j * wrapped_imag

    m = (1 << n_word)
    if n_word >= _n_word_max:
        dtype = object
        x = int_array(x).astype(dtype) & (m - 1)
    else:
        dtype = int
        x = np.array(x).astype(dtype) & (m - 1) 

    x = np.asarray(x).astype(dtype)

    if signed: 
        x = np.where(x < (1 << (n_word-1)), x, x | (-m))
        
    return x




_INT64_MIN = -(1 << 63)
_INT64_MAX = (1 << 63) - 1


def overflow_masks(x, val_min, val_max):
    """Return exact masks of values above `val_max` and below `val_min`.

    For float inputs, bounds wider than 53 bits are not representable in float64
    (e.g. float(2**62 - 1) == 2**62), so a plain comparison against the float
    bound could miss values beyond the limit. The comparison operator is chosen
    according to the rounding direction of the float bound, so masks are exact.
    NaN values are in neither mask.
    
    Parameters
    ---
    x : scalar or numpy.ndarray
        Candidate raw values (int, float or object dtype).
    val_min : int
        Minimum representable raw value.
    val_max : int
        Maximum representable raw value.
    
    Returns
    ---
    tuple of numpy.ndarray
        Boolean masks `(above, below)` with the shape of `x`."""
    x = np.asarray(x)

    if x.dtype.kind == 'f':
        f_max, f_min = float(val_max), float(val_min)
        with np.errstate(invalid='ignore'):
            above = (x >= f_max) if f_max > val_max else (x > f_max)
            below = (x <= f_min) if f_min < val_min else (x < f_min)
    else:
        # int and object (Python int) comparisons are exact
        above = x > val_max
        below = x < val_min

    return np.asarray(above, dtype=bool), np.asarray(below, dtype=bool)


def saturate(x, val_min, val_max, masks=None):
    """Saturate raw values to `[val_min, val_max]` with exact integer bounds.

    Vectorized replacement for `clip` in the overflow path. For float inputs,
    out-of-range positions get the bounds as integers, so bounds not representable
    in float64 are stored exactly; in-range values are truncated to int64 (same as
    the later cast to the raw dtype), or kept as Python objects when bounds do not
    fit int64. NaN saturates to `val_max` (legacy behavior).
    
    Parameters
    ---
    x : scalar or numpy.ndarray
        Candidate raw values (int, float or object dtype).
    val_min : int
        Minimum representable raw value.
    val_max : int
        Maximum representable raw value.
    masks : tuple of numpy.ndarray, optional
        Precomputed `(above, below)` masks from `overflow_masks`.
    
    Returns
    ---
    numpy.ndarray
        Saturated values."""
    x = np.asarray(x)

    if x.dtype == object:
        return np.asarray(np.clip(x, val_min, val_max))

    if x.dtype.kind in 'iu':
        info = np.iinfo(x.dtype)
        if info.min <= val_min and val_max <= info.max:
            # bounds cast to x dtype: avoids int/uint promotion to float (NumPy < 2)
            return np.clip(x, x.dtype.type(val_min), x.dtype.type(val_max))
        return np.asarray(np.clip(x.astype(object), val_min, val_max))

    above, below = overflow_masks(x, val_min, val_max) if masks is None else masks
    above = above | np.isnan(x)

    if _INT64_MIN <= val_min and val_max <= _INT64_MAX:
        y = np.where(above | below, 0, x).astype(np.int64)
    else:
        y = np.array(x, dtype=object)
    y[above] = val_max
    y[below] = val_min
    return y
