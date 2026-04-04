"""Numeric clipping/wrapping helpers for fixed-point internals."""

import warnings

import numpy as np

from .. import _n_word_max
from .common import int_array


_CLIP_KWARGS_WARNING_SHOWN = False


def _clip_scalar(x, val_min, val_max):
    """Clip a scalar using Python comparisons to preserve legacy semantics."""
    if val_min is None and val_max is None:
        raise ValueError("One of max or min must be given")
    if val_min is None:
        return np.array(min(val_max, x))
    if val_max is None:
        return np.array(max(val_min, x))
    return np.array(max(val_min, min(val_max, x)))


@np.vectorize
def clip_vectorized(x, val_min, val_max):
    """Legacy vectorized clip implementation.

    Parameters
    ---
    x : scalar, list, tuple or numpy.ndarray
        Input value(s) to clip.
    val_min : scalar or None
        Lower clipping bound.
    val_max : scalar or None
        Upper clipping bound.

    Returns
    ---
    scalar or numpy.ndarray
        Clipped value(s) within the provided bounds.
    """
    return _clip_scalar(x, val_min, val_max)


def _clip_can_use_numpy(x):
    """Return True when `np.clip` preserves the expected fxpmath contract.

    The fast path is intentionally narrow:
    - only ndarray inputs
    - no object dtype
    - no 0-D arrays/scalars

    Lists, tuples, scalars, and object arrays stay on the legacy path because
    they rely on Python integer semantics for very large values.
    """
    return (
        isinstance(x, np.ndarray)
        and x.ndim > 0
        and x.dtype != object
        and np.issubdtype(x.dtype, np.number)
    )


def _clip_numpy(x, val_min=None, val_max=None, **kwargs):
    """Call NumPy clip while preserving values outside a `where` mask."""
    if 'where' in kwargs and 'out' not in kwargs:
        kwargs = dict(kwargs)
        kwargs['out'] = np.array(x, copy=True)

    return np.clip(x, val_min, val_max, **kwargs)


def clip(x, val_min=None, val_max=None, **kwargs):
    """Clip fixed-point values using a fast NumPy path when it is safe.

    Parameters
    ---
    x : scalar, list, tuple or numpy.ndarray
        Input value(s) to clip.
    val_min : scalar or None, optional
        Lower clipping bound.
    val_max : scalar or None, optional
        Upper clipping bound.
    **kwargs : dict
        Optional NumPy clip kwargs forwarded only on the safe ndarray path.

    Returns
    ---
    scalar or numpy.ndarray
        Clipped value(s) constrained to the provided bounds.
    """
    global _CLIP_KWARGS_WARNING_SHOWN

    if _clip_can_use_numpy(x):
        return _clip_numpy(x, val_min, val_max, **kwargs)

    if kwargs and not _CLIP_KWARGS_WARNING_SHOWN:
        warnings.warn(
            "utils.clip ignored keyword arguments on the vectorized fallback path.",
            RuntimeWarning,
            stacklevel=2,
        )
        _CLIP_KWARGS_WARNING_SHOWN = True

    return clip_vectorized(x, val_min, val_max)


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
        Integer-clipped value(s) within bounds.
    """
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
        Wrapped integer value(s) constrained to fixed-point range.
    """
    if np.iscomplexobj(x):
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
