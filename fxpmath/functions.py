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
from .objects import Fxp, implements
from . import utils
from . import _n_word_max

try:
    from decimal import Decimal
except:
    Decimal = type(None)


def _get_sizing(vars, sizing, method, optimal_size=None):
        if not isinstance(vars, list):
            vars = [vars]

        signed = bool(np.any([v.signed for v in vars]))

        if sizing == 'optimal':
            if optimal_size is not None:
                signed, _, n_int, n_frac = optimal_size
            else:
                signed = vars[0].signed
                n_int = vars[0].n_int
                n_frac = vars[0].n_frac
        elif sizing == 'same':
            n_int = vars[0].n_int
            n_frac = vars[0].n_frac
        elif sizing == 'same_y':
            n_int = vars[-1].n_int
            n_frac = vars[-1].n_frac
        elif sizing == 'fit' and method == 'raw':
            n_int = None
            n_frac = max([v.n_frac for v in vars])
        elif sizing == 'fit' and method == 'repr':
            n_int = None
            n_frac = None
        elif sizing == 'largest':
            n_int = max([v.n_int for v in vars])
            n_frac = max([v.n_frac for v in vars])
        elif sizing == 'smallest':
            n_int = min([v.n_int for v in vars])
            n_frac = min([v.n_frac for v in vars])
        else:
            raise ValueError('{} is a wrong value for `sizing`. Valid values: optimal, same, fit, largest or smallest'.format(sizing))

        if n_frac is None or n_frac is None or n_int is None:
            n_word = None
        else:
            n_word = int(signed) + n_int + n_frac

        return signed, n_word, n_int, n_frac

def _function_over_one_var(repr_func, raw_func, x, out=None, out_like=None, sizing='optimal', method='raw', optimal_size=None, **kwargs):
    if not isinstance(x, Fxp):
        x = Fxp(x)

    signed, _, n_int, n_frac = _get_sizing([x], sizing=sizing, method=method, optimal_size=optimal_size)

    if out is not None:
        if isinstance(out, tuple):
            out = out[0] # recover only firts element
        if not isinstance(out, Fxp):
            raise TypeError('`out` must be a Fxp object!')
        if not out.signed and signed:
            raise ValueError('Signed addition can not be stored in unsigned `out` object!')
        n_frac = out.n_frac
        config = None

    elif out_like is not None:
        if not isinstance(out_like, Fxp):
            raise TypeError('`out_like` must be a Fxp object!')
        if not out_like.signed and signed:
            raise ValueError('Signed addition can not be stored in unsigned `out_like` object!')
        signed = None
        n_frac = None
        n_int = None
        config = None
    
    else:
        config = x.config

    if method == 'repr' or x.scaled or n_frac is None:
        raw = False
        val = repr_func(x.get_val(), **kwargs)
    elif method == 'raw':
        raw = True
        kwargs['n_frac'] = n_frac
        val = raw_func(x, **kwargs)
    else:
        raise ValueError('method {} is not valid. Valid methods: raw, repr'.format(method))

    if out is not None:
        z = out.set_val(val, raw=raw)
    else:
        z = Fxp(val, signed=signed, n_int=n_int, n_frac=n_frac, like=out_like, raw=raw)

    return z 

def _function_over_two_vars(repr_func, raw_func, x, y, out=None, out_like=None, sizing='optimal', method='raw', optimal_size=None, **kwargs):
    if not isinstance(x, Fxp):
        x = Fxp(x)
    if not isinstance(y, Fxp):
        y = Fxp(y)

    signed, _, n_int, n_frac = _get_sizing([x, y], sizing=sizing, method=method, optimal_size=optimal_size)

    if out is not None:
        if isinstance(out, tuple):
            out = out[0] # recover only firts element
        if not isinstance(out, Fxp):
            raise TypeError('`out` must be a Fxp object!')
        if not out.signed and signed:
            raise ValueError('Signed addition can not be stored in unsigned `out` object!')
        n_frac = out.n_frac
        config = None

    elif out_like is not None:
        if not isinstance(out_like, Fxp):
            raise TypeError('`out_like` must be a Fxp object!')
        if not out_like.signed and signed:
            raise ValueError('Signed addition can not be stored in unsigned `out_like` object!')
        signed = None
        n_frac = None
        n_int = None
        config = None

    else:
        config = x.config

    if method == 'repr' or x.scaled or n_frac is None:
        raw = False
        val = repr_func(x.get_val(), y.get_val(), **kwargs)
    elif method == 'raw':
        raw = True
        kwargs['n_frac'] = n_frac
        val = raw_func(x, y, **kwargs)
    else:
        raise ValueError('method {} is not valid. Valid methods: raw, repr'.format(method))

    if out is not None:
        z = out.set_val(val, raw=raw)
    else:
        z = Fxp(val, signed=signed, n_int=n_int, n_frac=n_frac, like=out_like, raw=raw, config=config)

    return z   

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

def fxp_sum(x, sizes='best_sizes', axis=None, dtype=None, out=None, vdtype=None):
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
    else:
        raise ValueError('Could not resolve output size!')

    return sum_along_axis

@implements(np.max)
def fxp_max(x, axis=None, out=None, out_like=None, sizing='optimal', method='raw', **kwargs):
    """
    """
    def _max_raw(x, n_frac, **kwargs):
        precision_cast = (lambda m: np.array(m, dtype=object)) if n_frac >= _n_word_max else (lambda m: m)
        return np.max(x.val, **kwargs) * precision_cast(2**(n_frac - x.n_frac))

    kwargs['axis'] = axis  
    return _function_over_one_var(repr_func=np.max, raw_func=_max_raw, x=x, out=out, out_like=out_like, sizing=sizing, method=method, **kwargs)

@implements(np.min)
def fxp_min(x, axis=None, out=None, out_like=None, sizing='optimal', method='raw', **kwargs):
    """
    """
    def _min_raw(x, n_frac, **kwargs):
        precision_cast = (lambda m: np.array(m, dtype=object)) if n_frac >= _n_word_max else (lambda m: m)
        return np.min(x.val, **kwargs) * precision_cast(2**(n_frac - x.n_frac))
    
    kwargs['axis'] = axis  
    return _function_over_one_var(repr_func=np.min, raw_func=_min_raw, x=x, out=out, out_like=out_like, sizing=sizing, method=method, **kwargs)

@implements(np.add)
def add(x, y, out=None, out_like=None, sizing='optimal', method='raw', **kwargs):
    """
    """
    def _add_raw(x, y, n_frac):
        precision_cast = (lambda m: np.array(m, dtype=object)) if n_frac >= _n_word_max else (lambda m: m)
        return x.val * precision_cast(2**(n_frac - x.n_frac)) + y.val * precision_cast(2**(n_frac - y.n_frac))

    if not isinstance(x, Fxp):
        x = Fxp(x)
    if not isinstance(y, Fxp):
        y = Fxp(y)

    signed = x.signed or y.signed
    n_int = max(x.n_int, y.n_int) + 1
    n_frac = max(x.n_frac, y.n_frac)
    n_word = int(signed) + n_int + n_frac
    optimal_size = (signed, n_word, n_int, n_frac)

    return _function_over_two_vars(repr_func=np.add, raw_func=_add_raw, x=x, y=y, out=out, out_like=out_like, sizing=sizing, method=method, optimal_size=optimal_size, **kwargs)

@implements(np.subtract)
def sub(x, y, out=None, out_like=None, sizing='optimal', method='raw', **kwargs):
    """
    """
    def _sub_raw(x, y, n_frac):
        precision_cast = (lambda m: np.array(m, dtype=object)) if n_frac >= _n_word_max else (lambda m: m)
        return x.val * precision_cast(2**(n_frac - x.n_frac)) - y.val * precision_cast(2**(n_frac - y.n_frac))

    if not isinstance(x, Fxp):
        x = Fxp(x)
    if not isinstance(y, Fxp):
        y = Fxp(y)

    signed = x.signed or y.signed
    n_int = max(x.n_int, y.n_int) + 1
    n_frac = max(x.n_frac, y.n_frac)
    n_word = int(signed) + n_int + n_frac
    optimal_size = (signed, n_word, n_int, n_frac)

    return _function_over_two_vars(repr_func=np.subtract, raw_func=_sub_raw, x=x, y=y, out=out, out_like=out_like, sizing=sizing, method=method, optimal_size=optimal_size, **kwargs)

@implements(np.multiply)
def mul(x, y, out=None, out_like=None, sizing='optimal', method='raw', **kwargs):
    """
    """
    def _mul_raw(x, y, n_frac):
        precision_cast = (lambda m: np.array(m, dtype=object)) if n_frac >= _n_word_max else (lambda m: m)
        raw_cast = (lambda m: np.array(m, dtype=object)) if (x.n_word + y.n_word) >= _n_word_max else (lambda m: m)
        return raw_cast(x.val) * raw_cast(y.val) * precision_cast(2**(n_frac - x.n_frac - y.n_frac))

    if not isinstance(x, Fxp):
        x = Fxp(x)
    if not isinstance(y, Fxp):
        y = Fxp(y)

    signed = x.signed or y.signed
    n_frac = x.n_frac + y.n_frac
    n_word = x.n_word + y.n_word
    n_int = n_word - int(signed) - n_frac
    optimal_size = (signed, n_word, n_int, n_frac)

    return _function_over_two_vars(repr_func=np.multiply, raw_func=_mul_raw, x=x, y=y, out=out, out_like=out_like, sizing=sizing, method=method, optimal_size=optimal_size, **kwargs)

@implements(np.floor_divide)
def floordiv(x, y, out=None, out_like=None, sizing='optimal', method='raw', **kwargs):
    """
    """
    def _floordiv_repr(x, y):
        return x // y
    def _floordiv_raw(x, y, n_frac):
        precision_cast = (lambda m: np.array(m, dtype=object)) if n_frac >= _n_word_max else (lambda m: m)
        return ((x.val * precision_cast(2**(n_frac - x.n_frac))) // (y.val * precision_cast(2**(n_frac - y.n_frac)))) * precision_cast(2**n_frac)

    if not isinstance(x, Fxp):
        x = Fxp(x)
    if not isinstance(y, Fxp):
        y = Fxp(y)

    signed = x.signed or y.signed
    n_int = x.n_int + y.n_frac + signed
    n_frac = 0
    n_word = int(signed) + n_int + n_frac
    optimal_size = (signed, n_word, n_int, n_frac)

    return _function_over_two_vars(repr_func=_floordiv_repr, raw_func=_floordiv_raw, x=x, y=y, out=out, out_like=out_like, sizing=sizing, method=method, optimal_size=optimal_size, **kwargs)

@implements(np.true_divide, np.divide)
def truediv(x, y, out=None, out_like=None, sizing='optimal', method='raw', **kwargs):
    """
    """
    def _truediv_repr(x, y):
        return x / y
    def _truediv_raw(x, y, n_frac):
        precision_cast = (lambda m: np.array(m, dtype=object)) if n_frac >= _n_word_max else (lambda m: m)
        return (x.val * precision_cast(2**(n_frac - x.n_frac + y.n_frac))) // y.val
        # return np.floor_divide(np.multiply(x.val, precision_cast(2**(n_frac - x.n_frac + y.n_frac))), y.val)

    if not isinstance(x, Fxp):
        x = Fxp(x)
    if not isinstance(y, Fxp):
        y = Fxp(y)

    signed = x.signed or y.signed
    n_int = x.n_int + y.n_frac + signed
    n_frac = x.n_frac + y.n_int
    n_word = int(signed) + n_int + n_frac
    optimal_size = (signed, n_word, n_int, n_frac)

    return _function_over_two_vars(repr_func=_truediv_repr, raw_func=_truediv_raw, x=x, y=y, out=out, out_like=out_like, sizing=sizing, method=method, optimal_size=optimal_size, **kwargs)

@implements(np.mod)
def mod(x, y, out=None, out_like=None, sizing='optimal', method='raw', **kwargs):
    """
    """
    def _mod_repr(x, y):
        return x % y
    def _mod_raw(x, y, n_frac):
        precision_cast = (lambda m: np.array(m, dtype=object)) if n_frac >= _n_word_max else (lambda m: m)
        return (x.val * precision_cast(2**(n_frac - x.n_frac))) % (y.val * precision_cast(2**(n_frac - y.n_frac)))

    if not isinstance(x, Fxp):
        x = Fxp(x)
    if not isinstance(y, Fxp):
        y = Fxp(y)

    signed = x.signed or y.signed
    n_int = max(x.n_int, y.n_int) if signed else min(x.n_int, y.n_int) # because python modulo implementation
    n_frac = max(x.n_frac, y.n_frac)
    n_word = int(signed) + n_int + n_frac
    optimal_size = (signed, n_word, n_int, n_frac)

    return _function_over_two_vars(repr_func=_mod_repr, raw_func=_mod_raw, x=x, y=y, out=out, out_like=out_like, sizing=sizing, method=method, optimal_size=optimal_size, **kwargs)

@implements(np.power)
def pow(x, y, out=None, out_like=None, sizing='optimal', method='raw', **kwargs):
    """
    """
    def _pow_repr(x, y):
        return x ** y

    def _pow_raw(x, y, n_frac):
        
        @np.vectorize
        def _power(x, y, x_n_frac, y_n_frac, n_frac):
            x_raw = int(x)
            y_raw = int(y)
            y_conv_factor = 2**y_n_frac
            _sign = 1

            if y_raw > 0:
                p1 = int(n_frac*y_conv_factor - y_raw*x_n_frac)
                if p1 >= 0:
                    z = (x_raw**y_raw) * (2**p1)
                else:
                    z = (x_raw**y_raw) // (2**(-p1))
            elif y_raw < 0:
                z = (2**(n_frac*y_conv_factor - y_raw*x_n_frac)) // (x_raw**(-1*y_raw))
            else:
                z = 2**n_frac
                y_conv_factor = 1 # force y_conv_factor
            
            if y_conv_factor != 1 and z != 0:
                z = z ** Decimal(1/y_conv_factor)
                _sign = int((x_raw/abs(x_raw))**(y_raw/y_conv_factor))

            return _sign*int(z)
        return _power(x.val, y.val, x.n_frac, y.n_frac, n_frac)  

    if not isinstance(x, Fxp):
        x = Fxp(x)
    if not isinstance(y, Fxp):
        y = Fxp(y)

    signed = x.signed or y.signed
    if y.n_frac == 0:
        if y.size == 1 and y.val >= 0:
            # non-negative integer exponent
            n_int = int(x.n_int * y.val + 1)
            n_frac = int(x.n_frac * y.val)
        elif y.size > 1 and np.all(y.val >= 0):
            # array of non-negative integer exponents
            n_int = int(x.n_int * np.max(y.val) + 1)
            n_frac = int(x.n_frac * np.max(y.val))
        else:
            # negative integer exponent
            n_int = n_frac = None # best sizes will be estimated
    else:
        # float exponent
        n_int = n_frac = None   # best sizes will be estimated
    if n_frac is not None:
        n_word = int(signed) + n_int + n_frac
    else:
        n_word = None
    optimal_size = (signed, n_word, n_int, n_frac)

    return _function_over_two_vars(repr_func=_pow_repr, raw_func=_pow_raw, x=x, y=y, out=out, out_like=out_like, sizing=sizing, method=method, optimal_size=optimal_size, **kwargs)

@implements(np.sum)
def sum(x, axis=None, out=None, out_like=None, sizing='optimal', method='raw', **kwargs):
    """
    """
    def _sum_raw(x, n_frac, **kwargs):
        precision_cast = (lambda m: np.array(m, dtype=object)) if n_frac >= _n_word_max else (lambda m: m)
        return np.sum(x.val, **kwargs) * precision_cast(2**(n_frac - x.n_frac))

    if not isinstance(x, Fxp):
        x = Fxp(x)

    signed = x.signed
    n_word = np.ceil(np.log2(x.size)).astype(int) + x.n_word
    n_frac = x.n_frac
    n_int = n_word - int(signed) - n_frac
    optimal_size = (signed, n_word, n_int, n_frac)

    kwargs['axis'] = axis
    return _function_over_one_var(repr_func=np.sum, raw_func=_sum_raw, x=x, out=out, out_like=out_like, sizing=sizing, method=method, optimal_size=optimal_size, **kwargs)

@implements(np.cumsum)
def cumsum(x, axis=None, out=None, out_like=None, sizing='optimal', method='raw', **kwargs):
    """
    """
    def _cumsum_raw(x, n_frac, **kwargs):
        precision_cast = (lambda m: np.array(m, dtype=object)) if n_frac >= _n_word_max else (lambda m: m)
        return np.cumsum(x.val, **kwargs) * precision_cast(2**(n_frac - x.n_frac))

    if not isinstance(x, Fxp):
        x = Fxp(x)

    signed = x.signed
    n_word = np.ceil(np.log2(x.size)).astype(int) + x.n_word
    n_frac = x.n_frac
    n_int = n_word - int(signed) - n_frac
    optimal_size = (signed, n_word, n_int, n_frac)

    kwargs['axis'] = axis
    return _function_over_one_var(repr_func=np.cumsum, raw_func=_cumsum_raw, x=x, out=out, out_like=out_like, sizing=sizing, method=method, optimal_size=optimal_size, **kwargs)

@implements(np.cumprod)
def cumprod(x, axis=None, out=None, out_like=None, sizing='optimal', method='raw', **kwargs):
    """
    """
    def _cumprod_raw(x, n_frac, **kwargs):
        axis = kwargs['axis'] if 'axis' in kwargs else None
        precision_cast = (lambda m: np.array(m, dtype=object)) if n_frac >= _n_word_max else (lambda m: m)
        pow_vals = n_frac - np.cumsum(np.ones_like(np.array(x)), axis=axis).astype(int)  * x.n_frac
        conv_factors = utils.int_array([2**pow_val for pow_val in precision_cast(pow_vals)])
        return np.cumprod(x.val, **kwargs) * conv_factors

    if not isinstance(x, Fxp):
        x = Fxp(x)

    signed = x.signed
    n_word = x.size * x.n_word
    n_frac = x.size * x.n_frac
    n_int = n_word - int(signed) - n_frac
    optimal_size = (signed, n_word, n_int, n_frac)

    kwargs['axis'] = axis
    return _function_over_one_var(repr_func=np.cumprod, raw_func=_cumprod_raw, x=x, out=out, out_like=out_like, sizing=sizing, method=method, optimal_size=optimal_size, **kwargs)

@implements(np.sort)
def sort(x, axis=-1, out=None, out_like=None, sizing='optimal', method='raw', **kwargs):
    """
    """
    def _sort_raw(x, n_frac, **kwargs):
        precision_cast = (lambda m: np.array(m, dtype=object)) if n_frac >= _n_word_max else (lambda m: m)
        return np.sort(x.val, **kwargs) * precision_cast(2**(n_frac - x.n_frac))

    kwargs['axis'] = axis
    return _function_over_one_var(repr_func=np.sort, raw_func=_sort_raw, x=x, out=out, out_like=out_like, sizing=sizing, method=method, **kwargs)

@implements(np.conjugate, np.conj)
def conjugate(x, out=None, out_like=None, sizing='optimal', method='raw', **kwargs):
    """
    """
    def _conjugate_raw(x, n_frac, **kwargs):
        precision_cast = (lambda m: np.array(m, dtype=object)) if n_frac >= _n_word_max else (lambda m: m)
        val_real = np.vectorize(lambda v: v.real)(x.val)
        val_imag = np.vectorize(lambda v: v.imag)(x.val)
        return (val_real -1j*val_imag) * precision_cast(2**(n_frac - x.n_frac))

    return _function_over_one_var(repr_func=np.conjugate, raw_func=_conjugate_raw, x=x, out=out, out_like=out_like, sizing=sizing, method=method, **kwargs)

@implements(np.transpose)
def transpose(x, axes=None, out=None, out_like=None, sizing='optimal', method='raw', **kwargs):
    """
    """
    def _transpose_raw(x, n_frac, **kwargs):
        precision_cast = (lambda m: np.array(m, dtype=object)) if n_frac >= _n_word_max else (lambda m: m)
        return (x.val.T) * precision_cast(2**(n_frac - x.n_frac))

    kwargs['axes'] = axes
    return _function_over_one_var(repr_func=np.transpose, raw_func=_transpose_raw, x=x, out=out, out_like=out_like, sizing=sizing, method=method, **kwargs)

@implements(np.clip)
def clip(a, a_min=None, a_max=None, out=None, out_like=None, sizing='optimal', method='raw', **kwargs):
    """
    """
    def _clip_raw(x, n_frac, **kwargs):
        precision_cast = (lambda m: np.array(m, dtype=object)) if n_frac >= _n_word_max else (lambda m: m)
        val_min = kwargs.pop('a_min', None)
        val_max = kwargs.pop('a_max', None)

        if val_min is not None: val_min *= 2**x.n_frac
        if val_max is not None: val_max *= 2**x.n_frac

        return utils.clip(x.val, val_min=val_min, val_max=val_max) * precision_cast(2**(n_frac - x.n_frac))

    kwargs['a_min'] = a_min
    kwargs['a_max'] = a_max
    return _function_over_one_var(repr_func=np.clip, raw_func=_clip_raw, x=a, out=out, out_like=out_like, sizing=sizing, method=method, **kwargs)

@implements(np.diagonal)
def diagonal(a, offset=0, axis1=0, axis2=1, out=None, out_like=None, sizing='optimal', method='raw', **kwargs):
    """
    """
    def _diagonal_raw(x, n_frac, **kwargs):
        precision_cast = (lambda m: np.array(m, dtype=object)) if n_frac >= _n_word_max else (lambda m: m)
        return np.diagonal(x.val, **kwargs) * precision_cast(2**(n_frac - x.n_frac))

    kwargs['offset'] = offset
    kwargs['axis1'] = axis1
    kwargs['axis2'] = axis2      
    return _function_over_one_var(repr_func=np.diagonal, raw_func=_diagonal_raw, x=a, out=out, out_like=out_like, sizing=sizing, method=method, **kwargs)

@implements(np.trace)
def trace(a, offset=0, axis1=0, axis2=1, out=None, out_like=None, sizing='optimal', method='raw', **kwargs):
    """
    """
    def _trace_raw(x, n_frac, **kwargs):
        precision_cast = (lambda m: np.array(m, dtype=object)) if n_frac >= _n_word_max else (lambda m: m)
        return np.trace(x.val, **kwargs) * precision_cast(2**(n_frac - x.n_frac))

    if not isinstance(a, Fxp):
        a = Fxp(a)

    num_of_additions = np.diagonal(np.array(a), offset=offset, axis1=axis1, axis2=axis2).size
    signed = a.signed
    n_word = np.ceil(np.log2(num_of_additions)).astype(int) + a.n_word
    n_frac = a.n_frac
    n_int = n_word - int(signed) - n_frac
    optimal_size = (signed, n_word, n_int, n_frac)

    kwargs['offset'] = offset
    kwargs['axis1'] = axis1
    kwargs['axis2'] = axis2      
    return _function_over_one_var(repr_func=np.trace, raw_func=_trace_raw, x=a, out=out, out_like=out_like, sizing=sizing, method=method, optimal_size=optimal_size, **kwargs)

@implements(np.prod)
def prod(a, axis=None, out=None, out_like=None, sizing='optimal', method='raw', **kwargs):
    """
    """
    def _prod_raw(x, n_frac, axis=None, **kwargs):
        precision_cast = (lambda m: np.array(m, dtype=object)) if n_frac >= _n_word_max else (lambda m: m)
        num_of_products = a.size if axis is None else a.shape[axis]
        return np.prod(x.val, axis=axis, **kwargs) * precision_cast(2**(n_frac - num_of_products * x.n_frac))

    if not isinstance(a, Fxp):
        a = Fxp(a)

    num_of_products = a.size if axis is None else a.shape[axis]
    signed = a.signed
    n_word = num_of_products * a.n_word
    n_frac = num_of_products * a.n_frac
    n_int = n_word - int(signed) - n_frac
    optimal_size = (signed, n_word, n_int, n_frac)

    kwargs['axis'] = axis  
    return _function_over_one_var(repr_func=np.prod, raw_func=_prod_raw, x=a, out=out, out_like=out_like, sizing=sizing, method=method, optimal_size=optimal_size, **kwargs)

@implements(np.dot)
def dot(x, y, out=None, out_like=None, sizing='optimal', method='raw', **kwargs):
    """
    """
    def _dot_raw(x, y, n_frac, **kwargs):
        precision_cast = (lambda m: np.array(m, dtype=object)) if n_frac >= _n_word_max else (lambda m: m)
        return np.dot(x.val, y.val, **kwargs) * precision_cast(2**(n_frac - x.n_frac - y.n_frac))

    if not isinstance(x, Fxp):
        x = Fxp(x)
    if not isinstance(y, Fxp):
        y = Fxp(y)

    num_of_additions = x.shape[-1]
    signed = x.signed or y.signed
    n_frac = x.n_frac + y.n_frac
    n_word = np.ceil(np.log2(num_of_additions)).astype(int) + x.n_word + y.n_word
    n_int = n_word - int(signed) - n_frac
    optimal_size = (signed, n_word, n_int, n_frac)

    return _function_over_two_vars(repr_func=np.dot, raw_func=_dot_raw, x=x, y=y, out=out, out_like=out_like, sizing=sizing, method=method, optimal_size=optimal_size, **kwargs)

@implements(np.nonzero)
def nonzero(x):
    """
    """
    if not isinstance(x, Fxp):
        x = Fxp(x)
    if x.scaled:
        return np.nonzero(x.get_val())
    else:
        return np.nonzero(x.val)