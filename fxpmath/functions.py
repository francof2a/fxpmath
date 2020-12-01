"""
fxpmath

---

A python library for fractional fixed-point arithmetic.

---

This software is provided under MIT License:

MIT License

Copyright (c) 2020 Franco, francof2a

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

#%% 
import numpy as np
from .objects import Fxp
from . import utils

def fxp_like(x, val=None):
    '''
    Returns a Fxp object like `x`.

    Parameters
    ---

    x : Fxp
        Object (Fxp) to copy.
    
    val : None or int or float or list or ndarray or str, optional, default=None
        Input value for the returned Fxp object.

    Returns
    ---

    y : Fxp
        New Fxp object like `x`.

    '''
    y = x.copy()
    return y(val)

def sum(x, sizes='best_sizes', axis=None, dtype=None, out=None, vdtype=None):
    '''
    Sum of array elements of a Fxp object, over a given axis.

    Paramters
    ---

    x : Fxp
        Elements to sum in a Fxp object.

    sizes : str, optional, default='best_sizes'
        Defines the returned Fxp sizes according input array size (val).
        * 'best_sizes': a extra word bit is added per couple of additions stage (log2(x().size))
        * 'tight_sizes': after calculate sum, the minimum sizes for n_word and n_frac are choosed.
        * 'same_sizes': same sizes than `x` are used to stored the result.

        If `dtype` or `out` are not None, `sizes` doesn't apply.

    axis : None or int or tuple of ints, optional, default=None
        Axis or axes along which a sum is performed. The default, axis=None, 
        will sum all of the elements of the input array. 
        If axis is negative it counts from the last to the first axis.

    dtype : str (Fxp dtype format), optional, default=None
        fxp-<sign><n_word>/<n_frac>-{complex}. i.e.: fxp-s16/15, fxp-u8/1, fxp-s32/24-complex
        If None, `sizes` or `out` are used to defined output format.

        A `dtype` can be alse extracted from a Fxp, i.e.: dtype=x.dtype

    out : Fxp, optional, default=None
        Alternative Fxp object to stored the result.
        If None, `sizes` or `dtype` are used to defined output format

    vdtype : dtype, optional, default=None
        The type of the returned array and of the accumulator in which the elements are summed.

    Returns
    ---
    sum_along_axis : Fxp
        A Fxp with an array with the same shape as `x` values, with the specified axis removed. 
        If `x` val is a 0-d array, or if axis is None, a scalar value is returned inside Fxp. 
        If an output array is specified, a reference to `out` is returned.

    '''
    if isinstance(x, Fxp):
        x_vals = x.get_val()
    else:
        x_vals = x

    x_sum = np.sum(x_vals, axis=axis, dtype=vdtype)

    if dtype is not None:
        signed, n_word, n_frac = utils.get_sizes_from_dtype(dtype)

        sum_along_axis = Fxp(x_sum, signed=signed, n_word=n_word, n_frac=n_frac)
    elif out is not None:
        if isinstance(out, Fxp):
            sum_along_axis = out(x_sum)
        else:
            raise TypeError('out argument must be a Fxp object!')
    elif sizes == 'best_sizes':
        signed = x.signed
        n_word = np.ceil(np.log2(x().size)).astype(int) + x.n_word
        n_frac = x.n_frac
        
        sum_along_axis = Fxp(x_sum, signed=signed, n_word=n_word, n_frac=n_frac)
    elif sizes == 'tight_sizes':
        sum_along_axis = Fxp(x_sum, signed=x.signed)
    elif sizes == 'same_sizes':
        sum_along_axis = Fxp(x_sum, like=x)

    return sum_along_axis
