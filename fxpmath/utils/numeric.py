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

