"""Shared utility helpers used across parse/representation/bitwise/numeric modules."""

import numpy as np
def array_support(func):
    """Decorate scalar helpers to operate element-wise on arrays.
    
    Parameters
    ---
    func : Callable
        Scalar helper function to decorate with array support.
    
    Returns
    ---
    Callable
        Decorator wrapper that adds recursive array handling to a scalar helper."""
    def iterator(*args, **kwargs):
        """Recursively apply the wrapped scalar helper element-wise over array-like inputs.
        
        Parameters
        ---
        *args : tuple
            Extra positional arguments forwarded to the wrapped callable.
        **kwargs : dict
            Extra keyword arguments forwarded to the underlying NumPy operation.
        
        Returns
        ---
        object
            Computed value."""
        if isinstance(args[0], (list, np.ndarray)) and np.asarray(args[0]).ndim > 0:
            vals = []
            for v in args[0]:
                vals.append(iterator(v, *args[1:], **kwargs))

            if isinstance(args[0], np.ndarray):
                vals = np.array(vals)
            return vals
        else:
            return func(*args, **kwargs)
    return iterator

def bits_len(x, signed=None):
    """Return the minimum number of bits required to represent the integer part of a value.
    
    Parameters
    ---
    x : int
        Integer value whose required bit width is measured.
    signed : bool, optional
        When `True`, include one sign bit in the returned length.
    
    Returns
    ---
    int
        Minimum bit width needed to represent the value."""
    if signed is None and x < 0:
        signed = True
    elif signed is None:
        signed = False
    elif not signed and x < 0:
        raise ValueError('negative value and unsigned type are incompatible!')

    n_bits = max( np.ceil(np.log2(np.abs(int(x)+0.5))).astype(int), 0) + signed
    return n_bits

def min_pow2(x, n_frac=0):
    """Return the smallest exponent `p` such that `2**p >= value` in magnitude.
    
    Parameters
    ---
    x : float
        Positive value used to compute floor(log2(x)).
    n_frac : int, optional
        Reference fractional scaling, used when values are already quantized.
    
    Returns
    ---
    int
        Greatest integer exponent `n` such that `2**n <= x`."""
    _pow = 1
    x = np.array(x)

    if np.any(x != 0):
        while not np.any(x % 2**_pow):
            _pow += 1
        _pow -= n_frac + 1 
    else:
        _pow = None
    
    return _pow

def int_array(x):
    """Convert inputs to integer ndarrays using safe dtypes for large values.
    
    Parameters
    ---
    x : scalar or array_like
        Input numeric values to cast to integer storage form.
    
    Returns
    ---
    int or numpy.ndarray
        Integer-cast scalar or array."""
    if not isinstance(x, np.ndarray):
        x = np.array(x)

    if x.dtype != complex:
        x = np.array(list(map(int, x.flatten()))).reshape(x.shape)
    else:
        x_real = np.vectorize(lambda v: v.real)(x)
        x_imag = np.vectorize(lambda v: v.imag)(x)
        x_real = np.array(list(map(int, x_real.flatten()))).reshape(x_real.shape)
        x_imag = np.array(list(map(int, x_imag.flatten()))).reshape(x_imag.shape)
        x = np.array(x_real + 1j*x_imag)
    return x