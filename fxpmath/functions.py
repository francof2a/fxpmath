"""fxpmath

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
SOFTWARE."""

#%% 
import numpy as np
from .objects import Fxp, implements
from . import utils
from .helpers import _cast_func, _use_object_cast

try:
    from decimal import Decimal
except:
    Decimal = type(None)


def _get_sizing(vars, sizing, method, optimal_size=None):
        """Resolve output signedness and size parameters for an operation.
        
        Parameters
        ---
        vars : list[Fxp] or Fxp
            Operand list used to infer output sizing.
        sizing : {'optimal', 'same', 'same_y', 'fit', 'largest', 'smallest'}
            Output sizing policy. `same_y` is used by right-hand/reflected operations.
        method : {'raw', 'repr'}
            Computation path: `raw` uses integer storage, `repr` uses represented numeric values.
        optimal_size : tuple[bool, int, int, int] or None, optional
            Explicit `(signed, n_word, n_int, n_frac)` sizing used when `sizing="optimal"`.
        
        Returns
        ---
        tuple[bool, int | None, int | None, int | None]
            Resolved output `(signed, n_word, n_int, n_frac)` sizing tuple."""
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
    """Apply a unary function over fixed-point inputs.
    
    Parameters
    ---
    repr_func : Callable
        Callable executed on represented values.
    raw_func : Callable
        Callable executed on raw integer storage.
    x : Fxp or array_like
        First operand or input value.
    out : Fxp, optional
        Destination fixed-point object used to store operation results.
    out_like : Fxp, optional
        Template fixed-point object used to construct output.
    sizing : {'optimal', 'same', 'same_y', 'fit', 'largest', 'smallest'}, optional
        Output sizing policy for fixed-point results.
    method : {'raw', 'repr'}, optional
        Computation path: `raw` uses integer storage; `repr` uses represented values.
    optimal_size : tuple[bool, int, int, int] or None, optional
        Explicit `(signed, n_word, n_int, n_frac)` tuple used when `sizing="optimal"`.
    **kwargs : dict
        Additional keyword arguments forwarded to the underlying NumPy function.
    
    Returns
    ---
    Fxp
        Fixed-point result produced by the unary kernel."""
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

    # propagate inaccuracy from argument
    if x.status['inaccuracy']:
        z.status['inaccuracy'] = True

    return z 

def _function_over_two_vars(repr_func, raw_func, x, y, out=None, out_like=None, sizing='optimal', method='raw', optimal_size=None, **kwargs):
    """Apply a binary function over fixed-point inputs.
    
    Parameters
    ---
    repr_func : Callable
        Callable executed on represented values.
    raw_func : Callable
        Callable executed on raw integer storage.
    x : Fxp or array_like
        First operand or input value.
    y : Fxp or array_like
        Second operand or input value.
    out : Fxp, optional
        Destination fixed-point object used to store operation results.
    out_like : Fxp, optional
        Template fixed-point object used to construct output.
    sizing : {'optimal', 'same', 'same_y', 'fit', 'largest', 'smallest'}, optional
        Output sizing policy for fixed-point results.
    method : {'raw', 'repr'}, optional
        Computation path: `raw` uses integer storage; `repr` uses represented values.
    optimal_size : tuple[bool, int, int, int] or None, optional
        Explicit `(signed, n_word, n_int, n_frac)` tuple used when `sizing="optimal"`.
    **kwargs : dict
        Additional keyword arguments forwarded to the underlying NumPy function.
    
    Returns
    ---
    Fxp
        Fixed-point result produced by the binary kernel."""
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

    # propagate inaccuracy from arguments
    if x.status['inaccuracy'] or y.status['inaccuracy']:
        z.status['inaccuracy'] = True

    return z   

def fxp_like(x, val=None):
    """Returns a Fxp object like `x`.
    
    Parameters
    ---
    
    x : Fxp
        Object (Fxp) to copy.
    
    val : None or int or float or list or ndarray or str, optional, default=None
        Input value for the returned Fxp object.
    
    Returns
    ---
    
    y : Fxp
        New Fxp object like `x`."""
    y = x.copy()
    return y(val)

def fxp_sum(x, sizes='best_sizes', axis=None, dtype=None, out=None, vdtype=None):
    """Sum of array elements of a Fxp object, over a given axis.
    
    Parameters
    ---
    
    x : Fxp
        Elements to sum in a Fxp object.
    
    sizes : str, optional, default='best_sizes'
        Defines the returned Fxp sizes according input array size (val).
        * 'best_sizes': a extra word bit is added per couple of additions stage (log2(x().size))
        * 'tight_sizes': after calculate sum, the minimum sizes for n_word and n_frac are chosen.
        * 'same_sizes': same sizes than `x` are used to stored the result.
    
        If `dtype` or `out` are not None, `sizes` doesn't apply.
    
    axis : None or int or tuple of ints, optional, default=None
        Axis or axes along which a sum is performed. The default, axis=None,
        will sum all of the elements of the input array.
        If axis is negative it counts from the last to the first axis.
    
    dtype : str (Fxp dtype format), optional, default=None
        fxp-<sign><n_word>/<n_frac>-{complex}. i.e.: fxp-s16/15, fxp-u8/1, fxp-s32/24-complex
        If None, `sizes` or `out` are used to defined output format.
    
        A `dtype` can be also extracted from a Fxp, i.e.: dtype=x.dtype
    
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
    
    Examples
    ---
    
    >>> from fxpmath import Fxp
    >>> import fxpmath.functions as fxp
    >>> x = Fxp([0.5, 1.5], signed=True, n_word=8, n_frac=4)
    >>> fxp.fxp_sum(x)()
    2.0"""
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
        n_word = int(np.ceil(np.log2(x().size))) + x.n_word
        n_frac = x.n_frac
        
        sum_along_axis = Fxp(x_sum, signed=signed, n_word=n_word, n_frac=n_frac)
    elif sizes == 'tight_sizes':
        sum_along_axis = Fxp(x_sum, signed=x.signed)
    elif sizes == 'same_sizes':
        sum_along_axis = Fxp(x_sum, like=x)
    else:
        raise ValueError('Could not resolve output size!')

    return sum_along_axis

def from_bin(x, **kwargs):
    """Create an `Fxp` object from a binary representation string.
    
    Parameters
    ---
    x : str
        Binary literal string (or collection of strings) accepted by `Fxp` input parsing.
    **kwargs : dict
        Keyword arguments forwarded to `Fxp(...)` (for example `signed`, `n_word`, `n_frac`, or `dtype`).
    
    Returns
    ---
    Fxp
        Fixed-point value parsed from binary string input.
    
    Examples
    ---
    >>> import fxpmath.functions as fxp
    >>> y = fxp.from_bin('0b0011.10', signed=False, n_word=8, n_frac=2)
    >>> y()
    3.5"""
    return Fxp(utils.add_binary_prefix(x), **kwargs)

@implements(np.max)
def fxp_max(x, axis=None, out=None, out_like=None, sizing='optimal', method='raw', **kwargs):
    """Return the maximum of an array or maximum along an axis.
    
    
    This function preserves NumPy semantics while honoring `Fxp` fixed-point sizing, rounding, overflow, and output-typing rules.
    
    Parameters
    ---
    x : Fxp or array_like
        First operand or input value.
    axis : int or tuple[int, ...], optional
        Axis or axes along which the operation is applied.
    out : Fxp, optional
        Destination fixed-point object used to store operation results.
    out_like : Fxp, optional
        Template fixed-point object used to construct output.
    sizing : {'optimal', 'same', 'same_y', 'fit', 'largest', 'smallest'}, optional
        Output sizing policy for fixed-point results.
    method : {'raw', 'repr'}, optional
        Computation path: `raw` uses integer storage; `repr` uses represented values.
    **kwargs : dict
        Additional keyword arguments forwarded to the underlying NumPy function.
    
    Returns
    ---
    Fxp or numpy.ndarray
        Operation result following `out`/`out_like` and configured output typing rules."""
    def _max_raw(x, n_frac, **kwargs):
        """Compute the maximum using raw integer storage and align the output fractional width.
        
        Parameters
        ---
        x : Fxp or array_like
            First operand or input value.
        n_frac : int
            Target fractional-bit width used for aligned raw arithmetic.
        **kwargs : dict
            Additional keyword arguments forwarded to the underlying NumPy function.
        
        Returns
        ---
        numpy.ndarray or scalar
            Intermediate raw- or represented-domain value returned by the helper kernel."""
        shift = n_frac - x.n_frac
        use_object = _use_object_cast(scale_terms=[(x.n_word, shift)])
        cast = _cast_func(use_object)
        return np.max(cast(x.val), **kwargs) * cast(2**shift)

    kwargs['axis'] = axis  
    return _function_over_one_var(repr_func=np.max, raw_func=_max_raw, x=x, out=out, out_like=out_like, sizing=sizing, method=method, **kwargs)

@implements(np.min)
def fxp_min(x, axis=None, out=None, out_like=None, sizing='optimal', method='raw', **kwargs):
    """Return the minimum of an array or minimum along an axis.
    
    
    This function preserves NumPy semantics while honoring `Fxp` fixed-point sizing, rounding, overflow, and output-typing rules.
    
    Parameters
    ---
    x : Fxp or array_like
        First operand or input value.
    axis : int or tuple[int, ...], optional
        Axis or axes along which the operation is applied.
    out : Fxp, optional
        Destination fixed-point object used to store operation results.
    out_like : Fxp, optional
        Template fixed-point object used to construct output.
    sizing : {'optimal', 'same', 'same_y', 'fit', 'largest', 'smallest'}, optional
        Output sizing policy for fixed-point results.
    method : {'raw', 'repr'}, optional
        Computation path: `raw` uses integer storage; `repr` uses represented values.
    **kwargs : dict
        Additional keyword arguments forwarded to the underlying NumPy function.
    
    Returns
    ---
    Fxp or numpy.ndarray
        Operation result following `out`/`out_like` and configured output typing rules."""
    def _min_raw(x, n_frac, **kwargs):
        """Compute the minimum using raw integer storage and align the output fractional width.
        
        Parameters
        ---
        x : Fxp or array_like
            First operand or input value.
        n_frac : int
            Target fractional-bit width used for aligned raw arithmetic.
        **kwargs : dict
            Additional keyword arguments forwarded to the underlying NumPy function.
        
        Returns
        ---
        numpy.ndarray or scalar
            Intermediate raw- or represented-domain value returned by the helper kernel."""
        shift = n_frac - x.n_frac
        use_object = _use_object_cast(scale_terms=[(x.n_word, shift)])
        cast = _cast_func(use_object)
        return np.min(cast(x.val), **kwargs) * cast(2**shift)
    
    kwargs['axis'] = axis  
    return _function_over_one_var(repr_func=np.min, raw_func=_min_raw, x=x, out=out, out_like=out_like, sizing=sizing, method=method, **kwargs)

@implements(np.add)
def add(x, y, out=None, out_like=None, sizing='optimal', method='raw', **kwargs):
    """Add arguments element-wise.
    
    
    This function preserves NumPy semantics while honoring `Fxp` fixed-point sizing, rounding, overflow, and output-typing rules.
    
    Parameters
    ---
    x : Fxp or array_like
        First operand or input value.
    y : Fxp or array_like
        Second operand or input value.
    out : Fxp, optional
        Destination fixed-point object used to store operation results.
    out_like : Fxp, optional
        Template fixed-point object used to construct output.
    sizing : {'optimal', 'same', 'same_y', 'fit', 'largest', 'smallest'}, optional
        Output sizing policy for fixed-point results.
    method : {'raw', 'repr'}, optional
        Computation path: `raw` uses integer storage; `repr` uses represented values.
    **kwargs : dict
        Additional keyword arguments forwarded to the underlying NumPy function.
    
    Returns
    ---
    Fxp or numpy.ndarray
        Operation result following `out`/`out_like` and configured output typing rules.
    
    Examples
    ---
    >>> from fxpmath import Fxp
    >>> import fxpmath.functions as fxp
    >>> a = Fxp(1.25, signed=True, n_word=8, n_frac=4)
    >>> b = Fxp(0.50, signed=True, n_word=8, n_frac=4)
    >>> fxp.add(a, b)()
    1.75"""
    def _add_raw(x, y, n_frac):
        """Add raw integer operands after aligning them to the requested fractional width.
        
        Parameters
        ---
        x : Fxp or array_like
            First operand or input value.
        y : Fxp or array_like
            Second operand or input value.
        n_frac : int
            Target fractional-bit width used for aligned raw arithmetic.
        
        Returns
        ---
        numpy.ndarray or scalar
            Intermediate raw- or represented-domain value returned by the helper kernel."""
        x_shift = n_frac - x.n_frac
        y_shift = n_frac - y.n_frac
        use_object = _use_object_cast(scale_terms=[(x.n_word, x_shift), (y.n_word, y_shift)])
        cast = _cast_func(use_object)
        return cast(x.val) * cast(2**x_shift) + cast(y.val) * cast(2**y_shift)

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
    """Subtract arguments, element-wise.
    
    
    This function preserves NumPy semantics while honoring `Fxp` fixed-point sizing, rounding, overflow, and output-typing rules.
    
    Parameters
    ---
    x : Fxp or array_like
        First operand or input value.
    y : Fxp or array_like
        Second operand or input value.
    out : Fxp, optional
        Destination fixed-point object used to store operation results.
    out_like : Fxp, optional
        Template fixed-point object used to construct output.
    sizing : {'optimal', 'same', 'same_y', 'fit', 'largest', 'smallest'}, optional
        Output sizing policy for fixed-point results.
    method : {'raw', 'repr'}, optional
        Computation path: `raw` uses integer storage; `repr` uses represented values.
    **kwargs : dict
        Additional keyword arguments forwarded to the underlying NumPy function.
    
    Returns
    ---
    Fxp or numpy.ndarray
        Operation result following `out`/`out_like` and configured output typing rules."""
    def _sub_raw(x, y, n_frac):
        """Subtract raw integer operands after aligning them to the requested fractional width.
        
        Parameters
        ---
        x : Fxp or array_like
            First operand or input value.
        y : Fxp or array_like
            Second operand or input value.
        n_frac : int
            Target fractional-bit width used for aligned raw arithmetic.
        
        Returns
        ---
        numpy.ndarray or scalar
            Intermediate raw- or represented-domain value returned by the helper kernel."""
        x_shift = n_frac - x.n_frac
        y_shift = n_frac - y.n_frac
        use_object = _use_object_cast(scale_terms=[(x.n_word, x_shift), (y.n_word, y_shift)])
        cast = _cast_func(use_object)
        return cast(x.val) * cast(2**x_shift) - cast(y.val) * cast(2**y_shift)

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
    """Multiply arguments element-wise.
    
    
    This function preserves NumPy semantics while honoring `Fxp` fixed-point sizing, rounding, overflow, and output-typing rules.
    
    Parameters
    ---
    x : Fxp or array_like
        First operand or input value.
    y : Fxp or array_like
        Second operand or input value.
    out : Fxp, optional
        Destination fixed-point object used to store operation results.
    out_like : Fxp, optional
        Template fixed-point object used to construct output.
    sizing : {'optimal', 'same', 'same_y', 'fit', 'largest', 'smallest'}, optional
        Output sizing policy for fixed-point results.
    method : {'raw', 'repr'}, optional
        Computation path: `raw` uses integer storage; `repr` uses represented values.
    **kwargs : dict
        Additional keyword arguments forwarded to the underlying NumPy function.
    
    Returns
    ---
    Fxp or numpy.ndarray
        Operation result following `out`/`out_like` and configured output typing rules."""
    def _mul_raw(x, y, n_frac):
        """Multiply raw integer operands and scale the product to the requested fractional width.
        
        Parameters
        ---
        x : Fxp or array_like
            First operand or input value.
        y : Fxp or array_like
            Second operand or input value.
        n_frac : int
            Target fractional-bit width used for aligned raw arithmetic.
        
        Returns
        ---
        numpy.ndarray or scalar
            Intermediate raw- or represented-domain value returned by the helper kernel."""
        shift = n_frac - x.n_frac - y.n_frac
        use_object = _use_object_cast(
            scale_terms=[(x.n_word + y.n_word, shift)],
            product_terms=[(x.n_word, y.n_word)]
        )
        cast = _cast_func(use_object)
        return cast(x.val) * cast(y.val) * cast(2**shift)

    if not isinstance(x, Fxp):
        x = Fxp(x)
    if not isinstance(y, Fxp):
        y = Fxp(y)

    is_complex = x.vdtype == complex or y.vdtype == complex

    signed = x.signed or y.signed
    n_frac = x.n_frac + y.n_frac
    n_word = x.n_word + y.n_word + int(is_complex)
    n_int = n_word - int(signed) - n_frac
    optimal_size = (signed, n_word, n_int, n_frac)

    return _function_over_two_vars(repr_func=np.multiply, raw_func=_mul_raw, x=x, y=y, out=out, out_like=out_like, sizing=sizing, method=method, optimal_size=optimal_size, **kwargs)

@implements(np.floor_divide)
def floordiv(x, y, out=None, out_like=None, sizing='optimal', method='raw', **kwargs):
    """Return the largest integer smaller or equal to the division of the inputs. It is equivalent to the Python ``//`` operator and pairs with the Python ``%`` (`remainder`), function so that ``a = a % b + b * (a // b)`` up to roundoff.
    
    
    This function preserves NumPy semantics while honoring `Fxp` fixed-point sizing, rounding, overflow, and output-typing rules.
    
    Parameters
    ---
    x : Fxp or array_like
        First operand or input value.
    y : Fxp or array_like
        Second operand or input value.
    out : Fxp, optional
        Destination fixed-point object used to store operation results.
    out_like : Fxp, optional
        Template fixed-point object used to construct output.
    sizing : {'optimal', 'same', 'same_y', 'fit', 'largest', 'smallest'}, optional
        Output sizing policy for fixed-point results.
    method : {'raw', 'repr'}, optional
        Computation path: `raw` uses integer storage; `repr` uses represented values.
    **kwargs : dict
        Additional keyword arguments forwarded to the underlying NumPy function.
    
    Returns
    ---
    Fxp or numpy.ndarray
        Operation result following `out`/`out_like` and configured output typing rules."""
    def _floordiv_repr(x, y):
        """Perform floor-division in represented-value space.
        
        Parameters
        ---
        x : Fxp or array_like
            First operand or input value.
        y : Fxp or array_like
            Second operand or input value.
        
        Returns
        ---
        numpy.ndarray or scalar
            Intermediate raw- or represented-domain value returned by the helper kernel."""
        return x // y

    def _floordiv_repr_complex(x, y):
        """Perform complex floor-division in represented-value space.
        
        Parameters
        ---
        x : Fxp or array_like
            First operand or input value.
        y : Fxp or array_like
            Second operand or input value.
        
        Returns
        ---
        numpy.ndarray or scalar
            Intermediate raw- or represented-domain value returned by the helper kernel."""
        y_norm = y.real ** 2 + y.imag ** 2
        real_part = (x.real * y.real + x.imag * y.imag) // y_norm
        imag_part = (x.imag * y.real - x.real * y.imag) // y_norm
        return real_part + 1j*imag_part
    
    def _floordiv_raw(x, y, n_frac):
        """Perform floor-division directly over raw integer storage.
        
        Parameters
        ---
        x : Fxp or array_like
            First operand or input value.
        y : Fxp or array_like
            Second operand or input value.
        n_frac : int
            Target fractional-bit width used for aligned raw arithmetic.
        
        Returns
        ---
        numpy.ndarray or scalar
            Intermediate raw- or represented-domain value returned by the helper kernel."""
        x_shift = n_frac - x.n_frac
        y_shift = n_frac - y.n_frac
        use_object = _use_object_cast(
            scale_terms=[(x.n_word, x_shift), (y.n_word, y_shift)],
            pow2_terms=[n_frac]
        )
        cast = _cast_func(use_object)
        return ((cast(x.val) * cast(2**x_shift)) // (cast(y.val) * cast(2**y_shift))) * cast(2**n_frac)

    def _floordiv_raw_complex(x, y, n_frac):
        """Perform complex floor-division directly over raw integer storage.
        
        Parameters
        ---
        x : Fxp or array_like
            First operand or input value.
        y : Fxp or array_like
            Second operand or input value.
        n_frac : int
            Target fractional-bit width used for aligned raw arithmetic.
        
        Returns
        ---
        numpy.ndarray or scalar
            Intermediate raw- or represented-domain value returned by the helper kernel."""
        norm_shift = n_frac - 2*y.n_frac
        num_shift = n_frac - x.n_frac - y.n_frac
        use_object = _use_object_cast(
            scale_terms=[(2*y.n_word, norm_shift), (x.n_word + y.n_word, num_shift)],
            product_terms=[(y.n_word, y.n_word), (x.n_word, y.n_word)],
            pow2_terms=[n_frac]
        )
        cast = _cast_func(use_object)

        y_norm = (cast(y.val.real) ** 2 + cast(y.val.imag) ** 2) * cast(2**norm_shift)
        real_part = (cast(x.val.real) * cast(y.val.real) + cast(x.val.imag) * cast(y.val.imag)) * cast(2**num_shift) // y_norm
        imag_part = (cast(x.val.imag) * cast(y.val.real) - cast(x.val.real) * cast(y.val.imag)) * cast(2**num_shift) // y_norm

        return (real_part + 1j*imag_part) * cast(2**n_frac)


    if not isinstance(x, Fxp):
        x = Fxp(x)
    if not isinstance(y, Fxp):
        y = Fxp(y)

    if x.vdtype == complex or y.vdtype == complex:
        _floordiv_repr = _floordiv_repr_complex
        _floordiv_raw = _floordiv_raw_complex

    signed = x.signed or y.signed
    n_int = x.n_int + y.n_frac + signed
    n_frac = 0
    n_word = int(signed) + n_int + n_frac
    optimal_size = (signed, n_word, n_int, n_frac)

    return _function_over_two_vars(repr_func=_floordiv_repr, raw_func=_floordiv_raw, x=x, y=y, out=out, out_like=out_like, sizing=sizing, method=method, optimal_size=optimal_size, **kwargs)

@implements(np.true_divide, np.divide)
def truediv(x, y, out=None, out_like=None, sizing='optimal', method='raw', **kwargs):
    """Divide arguments element-wise.
    
    
    This function preserves NumPy semantics while honoring `Fxp` fixed-point sizing, rounding, overflow, and output-typing rules.
    
    Parameters
    ---
    x : Fxp or array_like
        First operand or input value.
    y : Fxp or array_like
        Second operand or input value.
    out : Fxp, optional
        Destination fixed-point object used to store operation results.
    out_like : Fxp, optional
        Template fixed-point object used to construct output.
    sizing : {'optimal', 'same', 'same_y', 'fit', 'largest', 'smallest'}, optional
        Output sizing policy for fixed-point results.
    method : {'raw', 'repr'}, optional
        Computation path: `raw` uses integer storage; `repr` uses represented values.
    **kwargs : dict
        Additional keyword arguments forwarded to the underlying NumPy function.
    
    Returns
    ---
    Fxp or numpy.ndarray
        Operation result following `out`/`out_like` and configured output typing rules.
    
    Examples
    ---
    >>> from fxpmath import Fxp
    >>> import fxpmath.functions as fxp
    >>> a = Fxp(3.0, signed=True, n_word=16, n_frac=8)
    >>> b = Fxp(2.0, signed=True, n_word=16, n_frac=8)
    >>> fxp.truediv(a, b)()
    1.5"""
    def _truediv_repr(x, y):
        """Perform true-division in represented-value space.
        
        Parameters
        ---
        x : Fxp or array_like
            First operand or input value.
        y : Fxp or array_like
            Second operand or input value.
        
        Returns
        ---
        numpy.ndarray or scalar
            Intermediate raw- or represented-domain value returned by the helper kernel."""
        return x / y

    def _truediv_raw(x, y, n_frac):
        """Perform true-division directly over raw integer storage.
        
        Parameters
        ---
        x : Fxp or array_like
            First operand or input value.
        y : Fxp or array_like
            Second operand or input value.
        n_frac : int
            Target fractional-bit width used for aligned raw arithmetic.
        
        Returns
        ---
        numpy.ndarray or scalar
            Intermediate raw- or represented-domain value returned by the helper kernel."""
        shift = n_frac - x.n_frac + y.n_frac
        use_object = _use_object_cast(scale_terms=[(x.n_word, shift)])
        cast = _cast_func(use_object)
        return (cast(x.val) * cast(2**shift)) // cast(y.val)
        # return np.floor_divide(np.multiply(x.val, precision_cast(2**(n_frac - x.n_frac + y.n_frac))), y.val)

    def _truediv_raw_complex(x, y, n_frac):
        """Perform complex true-division directly over raw integer storage.
        
        Parameters
        ---
        x : Fxp or array_like
            First operand or input value.
        y : Fxp or array_like
            Second operand or input value.
        n_frac : int
            Target fractional-bit width used for aligned raw arithmetic.
        
        Returns
        ---
        numpy.ndarray or scalar
            Intermediate raw- or represented-domain value returned by the helper kernel."""
        shift = n_frac - x.n_frac + y.n_frac
        use_object = _use_object_cast(
            scale_terms=[(x.n_word + y.n_word, shift)],
            product_terms=[(y.n_word, y.n_word), (x.n_word, y.n_word)]
        )
        cast = _cast_func(use_object)

        y_norm = cast(y.val.real) ** 2 + cast(y.val.imag) ** 2
        real_part = (cast(x.val.real) * cast(y.val.real) + cast(x.val.imag) * cast(y.val.imag)) * cast(2**shift) // y_norm
        imag_part = (cast(x.val.imag) * cast(y.val.real) - cast(x.val.real) * cast(y.val.imag)) * cast(2**shift) // y_norm

        return real_part + 1j*imag_part


    if not isinstance(x, Fxp):
        x = Fxp(x)
    if not isinstance(y, Fxp):
        y = Fxp(y)

    if x.vdtype == complex or y.vdtype == complex:
        _truediv_raw = _truediv_raw_complex

    signed = x.signed or y.signed
    n_int = x.n_int + y.n_frac + signed
    n_frac = x.n_frac + y.n_int
    n_word = int(signed) + n_int + n_frac
    optimal_size = (signed, n_word, n_int, n_frac)

    return _function_over_two_vars(repr_func=_truediv_repr, raw_func=_truediv_raw, x=x, y=y, out=out, out_like=out_like, sizing=sizing, method=method, optimal_size=optimal_size, **kwargs)

@implements(np.mod)
def mod(x, y, out=None, out_like=None, sizing='optimal', method='raw', **kwargs):
    """Computes the remainder complementary to the `floor_divide` function.  It is equivalent to the Python modulus operator ``x1 % x2`` and has the same sign as the divisor `x2`. The MATLAB function equivalent to ``np.remainder`` is ``mod``.
    
    
    This function preserves NumPy semantics while honoring `Fxp` fixed-point sizing, rounding, overflow, and output-typing rules.
    
    Parameters
    ---
    x : Fxp or array_like
        First operand or input value.
    y : Fxp or array_like
        Second operand or input value.
    out : Fxp, optional
        Destination fixed-point object used to store operation results.
    out_like : Fxp, optional
        Template fixed-point object used to construct output.
    sizing : {'optimal', 'same', 'same_y', 'fit', 'largest', 'smallest'}, optional
        Output sizing policy for fixed-point results.
    method : {'raw', 'repr'}, optional
        Computation path: `raw` uses integer storage; `repr` uses represented values.
    **kwargs : dict
        Additional keyword arguments forwarded to the underlying NumPy function.
    
    Returns
    ---
    Fxp or numpy.ndarray
        Operation result following `out`/`out_like` and configured output typing rules."""
    def _mod_repr(x, y):
        """Compute modulo in represented-value space.
        
        Parameters
        ---
        x : Fxp or array_like
            First operand or input value.
        y : Fxp or array_like
            Second operand or input value.
        
        Returns
        ---
        numpy.ndarray or scalar
            Intermediate raw- or represented-domain value returned by the helper kernel."""
        return x % y
    def _mod_raw(x, y, n_frac):
        """Compute modulo directly over raw integer storage.
        
        Parameters
        ---
        x : Fxp or array_like
            First operand or input value.
        y : Fxp or array_like
            Second operand or input value.
        n_frac : int
            Target fractional-bit width used for aligned raw arithmetic.
        
        Returns
        ---
        numpy.ndarray or scalar
            Intermediate raw- or represented-domain value returned by the helper kernel."""
        x_shift = n_frac - x.n_frac
        y_shift = n_frac - y.n_frac
        use_object = _use_object_cast(scale_terms=[(x.n_word, x_shift), (y.n_word, y_shift)])
        cast = _cast_func(use_object)
        return (cast(x.val) * cast(2**x_shift)) % (cast(y.val) * cast(2**y_shift))

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
    """First array elements raised to powers from second array, element-wise.
    
    
    This function preserves NumPy semantics while honoring `Fxp` fixed-point sizing, rounding, overflow, and output-typing rules.
    
    Parameters
    ---
    x : Fxp or array_like
        First operand or input value.
    y : Fxp or array_like
        Second operand or input value.
    out : Fxp, optional
        Destination fixed-point object used to store operation results.
    out_like : Fxp, optional
        Template fixed-point object used to construct output.
    sizing : {'optimal', 'same', 'same_y', 'fit', 'largest', 'smallest'}, optional
        Output sizing policy for fixed-point results.
    method : {'raw', 'repr'}, optional
        Computation path: `raw` uses integer storage; `repr` uses represented values.
    **kwargs : dict
        Additional keyword arguments forwarded to the underlying NumPy function.
    
    Returns
    ---
    Fxp or numpy.ndarray
        Operation result following `out`/`out_like` and configured output typing rules."""
    def _pow_repr(x, y):
        """Compute exponentiation in represented-value space.
        
        Parameters
        ---
        x : Fxp or array_like
            First operand or input value.
        y : Fxp or array_like
            Second operand or input value.
        
        Returns
        ---
        numpy.ndarray or scalar
            Intermediate raw- or represented-domain value returned by the helper kernel."""
        return x ** y

    def _pow_raw(x, y, n_frac):
        
        """Compute exponentiation directly over raw integer storage.
        
        Parameters
        ---
        x : Fxp or array_like
            First operand or input value.
        y : Fxp or array_like
            Second operand or input value.
        n_frac : int
            Target fractional-bit width used for aligned raw arithmetic.
        
        Returns
        ---
        numpy.ndarray or scalar
            Intermediate raw- or represented-domain value returned by the helper kernel."""
        @np.vectorize
        def _power(x, y, x_n_frac, y_n_frac, n_frac):
            """Compute scalar exponentiation for vectorized raw power operations.
            
                    Parameters
            ---
                    x : Fxp or array_like
                        First operand or input value.
                    y : Fxp or array_like
                        Second operand or input value.
                    x_n_frac : int
                        Fractional-bit count for the base operand in raw exponentiation.
                    y_n_frac : int
                        Fractional-bit count for the exponent operand in raw exponentiation.
                    n_frac : int
                        Target fractional-bit width used for aligned raw arithmetic.
            
                    Returns
            ---
                    numpy.ndarray or scalar
                        Intermediate raw- or represented-domain value returned by the helper kernel."""
            x_raw = int(x)
            y_raw = int(y)
            x_n_frac = int(x_n_frac)
            y_n_frac = int(y_n_frac)
            n_frac = int(n_frac)
            y_conv_factor = int(2**y_n_frac)
            _sign = 1

            if y_raw > 0:
                p1 = int(n_frac*y_conv_factor - y_raw*x_n_frac)
                if p1 >= 0:
                    z = (x_raw**y_raw) * (2**p1)
                else:
                    z = (x_raw**y_raw) // (2**(-p1))
            elif y_raw < 0:
                p1 = int(n_frac*y_conv_factor - y_raw*x_n_frac)
                z = (2**p1) // (x_raw**(-1*y_raw))
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
    """Sum of array elements over a given axis.
    
    
    This function preserves NumPy semantics while honoring `Fxp` fixed-point sizing, rounding, overflow, and output-typing rules.
    
    Parameters
    ---
    x : Fxp or array_like
        First operand or input value.
    axis : int or tuple[int, ...], optional
        Axis or axes along which the operation is applied.
    out : Fxp, optional
        Destination fixed-point object used to store operation results.
    out_like : Fxp, optional
        Template fixed-point object used to construct output.
    sizing : {'optimal', 'same', 'same_y', 'fit', 'largest', 'smallest'}, optional
        Output sizing policy for fixed-point results.
    method : {'raw', 'repr'}, optional
        Computation path: `raw` uses integer storage; `repr` uses represented values.
    **kwargs : dict
        Additional keyword arguments forwarded to the underlying NumPy function.
    
    Returns
    ---
    Fxp or numpy.ndarray
        Operation result following `out`/`out_like` and configured output typing rules."""
    def _sum_raw(x, n_frac, **kwargs):
        """Compute summation over raw integer storage with fractional alignment.
        
        Parameters
        ---
        x : Fxp or array_like
            First operand or input value.
        n_frac : int
            Target fractional-bit width used for aligned raw arithmetic.
        **kwargs : dict
            Additional keyword arguments forwarded to the underlying NumPy function.
        
        Returns
        ---
        numpy.ndarray or scalar
            Intermediate raw- or represented-domain value returned by the helper kernel."""
        shift = n_frac - x.n_frac
        use_object = _use_object_cast(scale_terms=[(x.n_word, shift)])
        cast = _cast_func(use_object)
        return np.sum(cast(x.val), **kwargs) * cast(2**shift)

    if not isinstance(x, Fxp):
        x = Fxp(x)

    signed = x.signed
    n_word = int(np.ceil(np.log2(x.size))) + x.n_word
    n_frac = x.n_frac
    n_int = n_word - int(signed) - n_frac
    optimal_size = (signed, n_word, n_int, n_frac)

    kwargs['axis'] = axis
    return _function_over_one_var(repr_func=np.sum, raw_func=_sum_raw, x=x, out=out, out_like=out_like, sizing=sizing, method=method, optimal_size=optimal_size, **kwargs)

@implements(np.cumsum)
def cumsum(x, axis=None, out=None, out_like=None, sizing='optimal', method='raw', **kwargs):
    """Return the cumulative sum of the elements along a given axis.
    
    
    This function preserves NumPy semantics while honoring `Fxp` fixed-point sizing, rounding, overflow, and output-typing rules.
    
    Parameters
    ---
    x : Fxp or array_like
        First operand or input value.
    axis : int or tuple[int, ...], optional
        Axis or axes along which the operation is applied.
    out : Fxp, optional
        Destination fixed-point object used to store operation results.
    out_like : Fxp, optional
        Template fixed-point object used to construct output.
    sizing : {'optimal', 'same', 'same_y', 'fit', 'largest', 'smallest'}, optional
        Output sizing policy for fixed-point results.
    method : {'raw', 'repr'}, optional
        Computation path: `raw` uses integer storage; `repr` uses represented values.
    **kwargs : dict
        Additional keyword arguments forwarded to the underlying NumPy function.
    
    Returns
    ---
    Fxp or numpy.ndarray
        Operation result following `out`/`out_like` and configured output typing rules."""
    def _cumsum_raw(x, n_frac, **kwargs):
        """Compute cumulative summation over raw integer storage with fractional alignment.
        
        Parameters
        ---
        x : Fxp or array_like
            First operand or input value.
        n_frac : int
            Target fractional-bit width used for aligned raw arithmetic.
        **kwargs : dict
            Additional keyword arguments forwarded to the underlying NumPy function.
        
        Returns
        ---
        numpy.ndarray or scalar
            Intermediate raw- or represented-domain value returned by the helper kernel."""
        shift = n_frac - x.n_frac
        use_object = _use_object_cast(scale_terms=[(x.n_word, shift)])
        cast = _cast_func(use_object)
        return np.cumsum(cast(x.val), **kwargs) * cast(2**shift)

    if not isinstance(x, Fxp):
        x = Fxp(x)

    signed = x.signed
    n_word = int(np.ceil(np.log2(x.size))) + x.n_word
    n_frac = x.n_frac
    n_int = n_word - int(signed) - n_frac
    optimal_size = (signed, n_word, n_int, n_frac)

    kwargs['axis'] = axis
    return _function_over_one_var(repr_func=np.cumsum, raw_func=_cumsum_raw, x=x, out=out, out_like=out_like, sizing=sizing, method=method, optimal_size=optimal_size, **kwargs)

@implements(np.cumprod)
def cumprod(x, axis=None, out=None, out_like=None, sizing='optimal', method='raw', **kwargs):
    """Return the cumulative product of elements along a given axis.
    
    
    This function preserves NumPy semantics while honoring `Fxp` fixed-point sizing, rounding, overflow, and output-typing rules.
    
    Parameters
    ---
    x : Fxp or array_like
        First operand or input value.
    axis : int or tuple[int, ...], optional
        Axis or axes along which the operation is applied.
    out : Fxp, optional
        Destination fixed-point object used to store operation results.
    out_like : Fxp, optional
        Template fixed-point object used to construct output.
    sizing : {'optimal', 'same', 'same_y', 'fit', 'largest', 'smallest'}, optional
        Output sizing policy for fixed-point results.
    method : {'raw', 'repr'}, optional
        Computation path: `raw` uses integer storage; `repr` uses represented values.
    **kwargs : dict
        Additional keyword arguments forwarded to the underlying NumPy function.
    
    Returns
    ---
    Fxp or numpy.ndarray
        Operation result following `out`/`out_like` and configured output typing rules."""
    def _cumprod_raw(x, n_frac, **kwargs):
        """Compute cumulative products over raw integer storage with fractional alignment.
        
        Parameters
        ---
        x : Fxp or array_like
            First operand or input value.
        n_frac : int
            Target fractional-bit width used for aligned raw arithmetic.
        **kwargs : dict
            Additional keyword arguments forwarded to the underlying NumPy function.
        
        Returns
        ---
        numpy.ndarray or scalar
            Intermediate raw- or represented-domain value returned by the helper kernel."""
        axis = kwargs['axis'] if 'axis' in kwargs else None
        pow_vals = n_frac - np.cumsum(np.ones_like(np.array(x)), axis=axis).astype(int)  * x.n_frac
        max_pow = int(np.max(pow_vals)) if np.size(pow_vals) > 0 else 0
        use_object = _use_object_cast(
            scale_terms=[(x.n_word, max_pow)],
            product_terms=[x.size * x.n_word]
        )
        cast = _cast_func(use_object)
        conv_factors = np.array([2**int(pow_val) for pow_val in np.array(pow_vals).flatten()], dtype=object if use_object else None).reshape(np.shape(pow_vals))
        return np.cumprod(cast(x.val), **kwargs) * conv_factors

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
    """Return a sorted copy of an array.
    
    
    This function preserves NumPy semantics while honoring `Fxp` fixed-point sizing, rounding, overflow, and output-typing rules.
    
    Parameters
    ---
    x : Fxp or array_like
        First operand or input value.
    axis : int or tuple[int, ...], optional
        Axis or axes along which the operation is applied.
    out : Fxp, optional
        Destination fixed-point object used to store operation results.
    out_like : Fxp, optional
        Template fixed-point object used to construct output.
    sizing : {'optimal', 'same', 'same_y', 'fit', 'largest', 'smallest'}, optional
        Output sizing policy for fixed-point results.
    method : {'raw', 'repr'}, optional
        Computation path: `raw` uses integer storage; `repr` uses represented values.
    **kwargs : dict
        Additional keyword arguments forwarded to the underlying NumPy function.
    
    Returns
    ---
    Fxp or numpy.ndarray
        Operation result following `out`/`out_like` and configured output typing rules."""
    def _sort_raw(x, n_frac, **kwargs):
        """Sort raw integer values while preserving fixed-point scaling.
        
        Parameters
        ---
        x : Fxp or array_like
            First operand or input value.
        n_frac : int
            Target fractional-bit width used for aligned raw arithmetic.
        **kwargs : dict
            Additional keyword arguments forwarded to the underlying NumPy function.
        
        Returns
        ---
        numpy.ndarray or scalar
            Intermediate raw- or represented-domain value returned by the helper kernel."""
        shift = n_frac - x.n_frac
        use_object = _use_object_cast(scale_terms=[(x.n_word, shift)])
        cast = _cast_func(use_object)
        return np.sort(cast(x.val), **kwargs) * cast(2**shift)

    kwargs['axis'] = axis
    return _function_over_one_var(repr_func=np.sort, raw_func=_sort_raw, x=x, out=out, out_like=out_like, sizing=sizing, method=method, **kwargs)

@implements(np.conjugate, np.conj)
def conjugate(x, out=None, out_like=None, sizing='optimal', method='raw', **kwargs):
    """Return the complex conjugate, element-wise.
    
    
    This function preserves NumPy semantics while honoring `Fxp` fixed-point sizing, rounding, overflow, and output-typing rules.
    
    Parameters
    ---
    x : Fxp or array_like
        First operand or input value.
    out : Fxp, optional
        Destination fixed-point object used to store operation results.
    out_like : Fxp, optional
        Template fixed-point object used to construct output.
    sizing : {'optimal', 'same', 'same_y', 'fit', 'largest', 'smallest'}, optional
        Output sizing policy for fixed-point results.
    method : {'raw', 'repr'}, optional
        Computation path: `raw` uses integer storage; `repr` uses represented values.
    **kwargs : dict
        Additional keyword arguments forwarded to the underlying NumPy function.
    
    Returns
    ---
    Fxp or numpy.ndarray
        Operation result following `out`/`out_like` and configured output typing rules."""
    def _conjugate_raw(x, n_frac, **kwargs):
        """Compute complex conjugates from raw integer real and imaginary parts.
        
        Parameters
        ---
        x : Fxp or array_like
            First operand or input value.
        n_frac : int
            Target fractional-bit width used for aligned raw arithmetic.
        **kwargs : dict
            Additional keyword arguments forwarded to the underlying NumPy function.
        
        Returns
        ---
        numpy.ndarray or scalar
            Intermediate raw- or represented-domain value returned by the helper kernel."""
        shift = n_frac - x.n_frac
        use_object = _use_object_cast(scale_terms=[(x.n_word, shift)])
        cast = _cast_func(use_object)
        val_real = np.vectorize(lambda v: v.real)(x.val)
        val_imag = np.vectorize(lambda v: v.imag)(x.val)
        return (cast(val_real) -1j*cast(val_imag)) * cast(2**shift)

    return _function_over_one_var(repr_func=np.conjugate, raw_func=_conjugate_raw, x=x, out=out, out_like=out_like, sizing=sizing, method=method, **kwargs)

@implements(np.transpose)
def transpose(x, axes=None, out=None, out_like=None, sizing='optimal', method='raw', **kwargs):
    """For a 1-D array, this returns an unchanged view of the original array, as a transposed vector is simply the same vector. To convert a 1-D array into a 2-D column vector, an additional dimension must be added, e.g., ``np.atleast_2d(a).T`` achieves this, as does ``a[:, np.newaxis]``. For a 2-D array, this is the standard matrix transpose. For an n-D array, if axes are given, their order indicates how the axes are permuted (see Examples). If axes are not provided, then ``transpose(a).shape == a.shape[::-1]``.
    
    
    This function preserves NumPy semantics while honoring `Fxp` fixed-point sizing, rounding, overflow, and output-typing rules.
    
    Parameters
    ---
    x : Fxp or array_like
        First operand or input value.
    axes : tuple[int, ...], optional
        Axis permutation used by transpose-style operations.
    out : Fxp, optional
        Destination fixed-point object used to store operation results.
    out_like : Fxp, optional
        Template fixed-point object used to construct output.
    sizing : {'optimal', 'same', 'same_y', 'fit', 'largest', 'smallest'}, optional
        Output sizing policy for fixed-point results.
    method : {'raw', 'repr'}, optional
        Computation path: `raw` uses integer storage; `repr` uses represented values.
    **kwargs : dict
        Additional keyword arguments forwarded to the underlying NumPy function.
    
    Returns
    ---
    Fxp or numpy.ndarray
        Operation result following `out`/`out_like` and configured output typing rules."""
    def _transpose_raw(x, n_frac, **kwargs):
        """Transpose raw integer storage while preserving fixed-point scaling.
        
        Parameters
        ---
        x : Fxp or array_like
            First operand or input value.
        n_frac : int
            Target fractional-bit width used for aligned raw arithmetic.
        **kwargs : dict
            Additional keyword arguments forwarded to the underlying NumPy function.
        
        Returns
        ---
        numpy.ndarray or scalar
            Intermediate raw- or represented-domain value returned by the helper kernel."""
        shift = n_frac - x.n_frac
        use_object = _use_object_cast(scale_terms=[(x.n_word, shift)])
        cast = _cast_func(use_object)
        return cast(x.val.T) * cast(2**shift)

    kwargs['axes'] = axes
    return _function_over_one_var(repr_func=np.transpose, raw_func=_transpose_raw, x=x, out=out, out_like=out_like, sizing=sizing, method=method, **kwargs)

@implements(np.clip)
def clip(a, a_min=None, a_max=None, out=None, out_like=None, sizing='optimal', method='raw', **kwargs):
    """Clip (limit) the values in an array.
    
        This function preserves NumPy semantics while honoring `Fxp` fixed-point sizing, rounding, overflow, and output-typing rules.
    
    Parameters
    ---
    a : Fxp or array_like
        Input array or scalar values.
    a_min : scalar or None, optional
        Lower bound. Values below this limit are clipped.
    a_max : scalar or None, optional
        Upper bound. Values above this limit are clipped.
    out : Fxp, optional
        Destination fixed-point container where results are written.
    out_like : Fxp, optional
        Template fixed-point object used to build the output container.
    sizing : {'optimal', 'same', 'same_y', 'fit', 'largest', 'smallest'}, optional
        Output sizing policy. `same_y` is used by right-hand/reflected operations.
    method : {'raw', 'repr'}, optional
        Computation path: `raw` uses integer storage, `repr` uses represented numeric values.
    **kwargs : dict
        Extra keyword arguments forwarded to the underlying NumPy operation.
    
    Returns
    ---
    Fxp or numpy.ndarray
        Operation result following `out`/`out_like` and array output configuration rules."""
    def _clip_raw(x, n_frac, **kwargs):
        """Clip raw integer values to the requested bounds.
        
        Parameters
        ---
        x : Fxp or array_like
            First operand or input value.
        n_frac : int
            Target fractional-bit width used for aligned raw arithmetic.
        **kwargs : dict
            Additional keyword arguments forwarded to the underlying NumPy function.
        
        Returns
        ---
        numpy.ndarray or scalar
            Intermediate raw- or represented-domain value returned by the helper kernel."""
        shift = n_frac - x.n_frac
        use_object = _use_object_cast(scale_terms=[(x.n_word, shift)])
        cast = _cast_func(use_object)
        val_min = kwargs.pop('a_min', None)
        val_max = kwargs.pop('a_max', None)

        if val_min is not None: val_min *= 2**x.n_frac
        if val_max is not None: val_max *= 2**x.n_frac

        return cast(utils.clip(cast(x.val), val_min=val_min, val_max=val_max)) * cast(2**shift)

    kwargs['a_min'] = a_min
    kwargs['a_max'] = a_max
    return _function_over_one_var(repr_func=np.clip, raw_func=_clip_raw, x=a, out=out, out_like=out_like, sizing=sizing, method=method, **kwargs)

@implements(np.diagonal)
def diagonal(a, offset=0, axis1=0, axis2=1, out=None, out_like=None, sizing='optimal', method='raw', **kwargs):
    """Return specified diagonals.
    
        This function preserves NumPy semantics while honoring `Fxp` fixed-point sizing, rounding, overflow, and output-typing rules.
    
    Parameters
    ---
    a : Fxp or array_like
        Input array or scalar values.
    offset : int, optional
        Diagonal offset from the main diagonal.
    axis1 : int, optional
        First axis used to define matrix diagonals.
    axis2 : int, optional
        Second axis used to define matrix diagonals.
    out : Fxp, optional
        Destination fixed-point container where results are written.
    out_like : Fxp, optional
        Template fixed-point object used to build the output container.
    sizing : {'optimal', 'same', 'same_y', 'fit', 'largest', 'smallest'}, optional
        Output sizing policy. `same_y` is used by right-hand/reflected operations.
    method : {'raw', 'repr'}, optional
        Computation path: `raw` uses integer storage, `repr` uses represented numeric values.
    **kwargs : dict
        Extra keyword arguments forwarded to the underlying NumPy operation.
    
    Returns
    ---
    Fxp or numpy.ndarray
        Operation result following `out`/`out_like` and array output configuration rules."""
    def _diagonal_raw(x, n_frac, **kwargs):
        """Extract diagonal values from raw integer storage.
        
        Parameters
        ---
        x : Fxp or array_like
            First operand or input value.
        n_frac : int
            Target fractional-bit width used for aligned raw arithmetic.
        **kwargs : dict
            Additional keyword arguments forwarded to the underlying NumPy function.
        
        Returns
        ---
        numpy.ndarray or scalar
            Intermediate raw- or represented-domain value returned by the helper kernel."""
        shift = n_frac - x.n_frac
        use_object = _use_object_cast(scale_terms=[(x.n_word, shift)])
        cast = _cast_func(use_object)
        return np.diagonal(cast(x.val), **kwargs) * cast(2**shift)

    kwargs['offset'] = offset
    kwargs['axis1'] = axis1
    kwargs['axis2'] = axis2      
    return _function_over_one_var(repr_func=np.diagonal, raw_func=_diagonal_raw, x=a, out=out, out_like=out_like, sizing=sizing, method=method, **kwargs)

@implements(np.trace)
def trace(a, offset=0, axis1=0, axis2=1, out=None, out_like=None, sizing='optimal', method='raw', **kwargs):
    """Return the sum along diagonals of the array.
    
        This function preserves NumPy semantics while honoring `Fxp` fixed-point sizing, rounding, overflow, and output-typing rules.
    
    Parameters
    ---
    a : Fxp or array_like
        Input array or scalar values.
    offset : int, optional
        Diagonal offset from the main diagonal.
    axis1 : int, optional
        First axis used to define matrix diagonals.
    axis2 : int, optional
        Second axis used to define matrix diagonals.
    out : Fxp, optional
        Destination fixed-point container where results are written.
    out_like : Fxp, optional
        Template fixed-point object used to build the output container.
    sizing : {'optimal', 'same', 'same_y', 'fit', 'largest', 'smallest'}, optional
        Output sizing policy. `same_y` is used by right-hand/reflected operations.
    method : {'raw', 'repr'}, optional
        Computation path: `raw` uses integer storage, `repr` uses represented numeric values.
    **kwargs : dict
        Extra keyword arguments forwarded to the underlying NumPy operation.
    
    Returns
    ---
    Fxp or numpy.ndarray
        Operation result following `out`/`out_like` and array output configuration rules."""
    def _trace_raw(x, n_frac, **kwargs):
        """Compute the trace from raw integer storage.
        
        Parameters
        ---
        x : Fxp or array_like
            First operand or input value.
        n_frac : int
            Target fractional-bit width used for aligned raw arithmetic.
        **kwargs : dict
            Additional keyword arguments forwarded to the underlying NumPy function.
        
        Returns
        ---
        numpy.ndarray or scalar
            Intermediate raw- or represented-domain value returned by the helper kernel."""
        shift = n_frac - x.n_frac
        use_object = _use_object_cast(scale_terms=[(x.n_word, shift)])
        cast = _cast_func(use_object)
        return np.trace(cast(x.val), **kwargs) * cast(2**shift)

    if not isinstance(a, Fxp):
        a = Fxp(a)

    num_of_additions = np.diagonal(np.array(a), offset=offset, axis1=axis1, axis2=axis2).size
    signed = a.signed
    n_word = int(np.ceil(np.log2(num_of_additions))) + a.n_word
    n_frac = a.n_frac
    n_int = n_word - int(signed) - n_frac
    optimal_size = (signed, n_word, n_int, n_frac)

    kwargs['offset'] = offset
    kwargs['axis1'] = axis1
    kwargs['axis2'] = axis2      
    return _function_over_one_var(repr_func=np.trace, raw_func=_trace_raw, x=a, out=out, out_like=out_like, sizing=sizing, method=method, optimal_size=optimal_size, **kwargs)

@implements(np.prod)
def prod(a, axis=None, out=None, out_like=None, sizing='optimal', method='raw', **kwargs):
    """Return the product of array elements over a given axis.
    
        This function preserves NumPy semantics while honoring `Fxp` fixed-point sizing, rounding, overflow, and output-typing rules.
    
    Parameters
    ---
    a : Fxp or array_like
        Input array or scalar values.
    axis : int or tuple[int, ...], optional
        Axis or axes along which the operation is applied.
    out : Fxp, optional
        Destination fixed-point container where results are written.
    out_like : Fxp, optional
        Template fixed-point object used to build the output container.
    sizing : {'optimal', 'same', 'same_y', 'fit', 'largest', 'smallest'}, optional
        Output sizing policy. `same_y` is used by right-hand/reflected operations.
    method : {'raw', 'repr'}, optional
        Computation path: `raw` uses integer storage, `repr` uses represented numeric values.
    **kwargs : dict
        Extra keyword arguments forwarded to the underlying NumPy operation.
    
    Returns
    ---
    Fxp or numpy.ndarray
        Operation result following `out`/`out_like` and array output configuration rules."""
    def _prod_raw(x, n_frac, axis=None, **kwargs):
        """Compute multiplicative reduction over raw integer storage.
        
        Parameters
        ---
        x : Fxp or array_like
            First operand or input value.
        n_frac : int
            Target fractional-bit width used for aligned raw arithmetic.
        axis : int or tuple[int, ...], optional
            Axis or axes along which the operation is applied.
        **kwargs : dict
            Additional keyword arguments forwarded to the underlying NumPy function.
        
        Returns
        ---
        numpy.ndarray or scalar
            Intermediate raw- or represented-domain value returned by the helper kernel."""
        num_of_products = a.size if axis is None else a.shape[axis]
        shift = n_frac - num_of_products * x.n_frac
        use_object = _use_object_cast(
            scale_terms=[(num_of_products * x.n_word, shift)],
            product_terms=[num_of_products * x.n_word]
        )
        cast = _cast_func(use_object)
        return np.prod(cast(x.val), axis=axis, **kwargs) * cast(2**shift)

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
    """Compute the dot product of two arrays.
    
    
    This function preserves NumPy semantics while honoring `Fxp` fixed-point sizing, rounding, overflow, and output-typing rules.
    
    Parameters
    ---
    x : Fxp or array_like
        First operand or input value.
    y : Fxp or array_like
        Second operand or input value.
    out : Fxp, optional
        Destination fixed-point object used to store operation results.
    out_like : Fxp, optional
        Template fixed-point object used to construct output.
    sizing : {'optimal', 'same', 'same_y', 'fit', 'largest', 'smallest'}, optional
        Output sizing policy for fixed-point results.
    method : {'raw', 'repr'}, optional
        Computation path: `raw` uses integer storage; `repr` uses represented values.
    **kwargs : dict
        Additional keyword arguments forwarded to the underlying NumPy function.
    
    Returns
    ---
    Fxp or numpy.ndarray
        Operation result following `out`/`out_like` and configured output typing rules."""
    def _dot_raw(x, y, n_frac, **kwargs):
        """Compute dot products over raw integer storage.
        
        Parameters
        ---
        x : Fxp or array_like
            First operand or input value.
        y : Fxp or array_like
            Second operand or input value.
        n_frac : int
            Target fractional-bit width used for aligned raw arithmetic.
        **kwargs : dict
            Additional keyword arguments forwarded to the underlying NumPy function.
        
        Returns
        ---
        numpy.ndarray or scalar
            Intermediate raw- or represented-domain value returned by the helper kernel."""
        shift = n_frac - x.n_frac - y.n_frac
        use_object = _use_object_cast(
            scale_terms=[(x.n_word + y.n_word, shift)],
            product_terms=[(x.n_word, y.n_word)]
        )
        cast = _cast_func(use_object)
        return np.dot(cast(x.val), cast(y.val), **kwargs) * cast(2**shift)

    if not isinstance(x, Fxp):
        x = Fxp(x)
    if not isinstance(y, Fxp):
        y = Fxp(y)

    num_of_additions = x.shape[-1]
    signed = x.signed or y.signed
    n_frac = x.n_frac + y.n_frac
    n_word = int(np.ceil(np.log2(num_of_additions))) + x.n_word + y.n_word
    n_int = n_word - int(signed) - n_frac
    optimal_size = (signed, n_word, n_int, n_frac)

    return _function_over_two_vars(repr_func=np.dot, raw_func=_dot_raw, x=x, y=y, out=out, out_like=out_like, sizing=sizing, method=method, optimal_size=optimal_size, **kwargs)

@implements(np.nonzero)
def nonzero(x):
    """Return the indices of the elements that are non-zero.
    
    
    This function preserves NumPy semantics while honoring `Fxp` fixed-point sizing, rounding, overflow, and output-typing rules.
    
    Parameters
    ---
    x : Fxp or array_like
        First operand or input value.
    
    Returns
    ---
    Fxp or numpy.ndarray
        Operation result following `out`/`out_like` and configured output typing rules."""
    if not isinstance(x, Fxp):
        x = Fxp(x)
    if x.scaled:
        return np.nonzero(x.get_val())
    else:
        return np.nonzero(x.val)
    
@implements(np.reshape)
def reshape(a, shape=None, order='C', out=None, out_like=None, sizing='same', method='raw', **kwargs):
    """Gives a new shape to an array without changing its data.
    
    
    This function preserves NumPy semantics while honoring `Fxp` fixed-point sizing, rounding, overflow, and output-typing rules.
    
    Parameters
    ---
    a : Fxp or array_like
        Input array or scalar values.
    shape : int or tuple[int, ...], optional
        Target output shape.
    order : {'C', 'F', 'A', 'K'}, optional
        Index order used by reshape operations.
    out : Fxp, optional
        Destination fixed-point object used to store operation results.
    out_like : Fxp, optional
        Template fixed-point object used to construct output.
    sizing : {'optimal', 'same', 'same_y', 'fit', 'largest', 'smallest'}, optional
        Output sizing policy for fixed-point results.
    method : {'raw', 'repr'}, optional
        Computation path: `raw` uses integer storage; `repr` uses represented values.
    **kwargs : dict
        Additional keyword arguments forwarded to the underlying NumPy function.
    
    Returns
    ---
    Fxp or numpy.ndarray
        Operation result following `out`/`out_like` and configured output typing rules."""
    # compatibility alias for callers still using `newshape=...`
    newshape = kwargs.pop('newshape', None)
    if shape is None:
        shape = newshape
    elif newshape is not None and shape != newshape:
        raise TypeError('`shape` and `newshape` can not be different values!')

    if shape is None:
        raise TypeError("reshape() missing 1 required argument: 'shape'")

    def _reshape_repr(x, shape, order, **kwargs):
        """Reshape represented values using NumPy reshape semantics.
        
        Parameters
        ---
        x : Fxp or array_like
            First operand or input value.
        shape : int or tuple[int, ...]
            Target output shape.
        order : {'C', 'F', 'A', 'K'}
            Index order used by reshape operations.
        **kwargs : dict
            Additional keyword arguments forwarded to the underlying NumPy function.
        
        Returns
        ---
        numpy.ndarray or scalar
            Intermediate raw- or represented-domain value returned by the helper kernel."""
        return np.reshape(x, shape, order=order)

    def _reshape_raw(x, shape, order, **kwargs):
        """Reshape raw integer storage using NumPy reshape semantics.
        
        Parameters
        ---
        x : Fxp or array_like
            First operand or input value.
        shape : int or tuple[int, ...]
            Target output shape.
        order : {'C', 'F', 'A', 'K'}
            Index order used by reshape operations.
        **kwargs : dict
            Additional keyword arguments forwarded to the underlying NumPy function.
        
        Returns
        ---
        numpy.ndarray or scalar
            Intermediate raw- or represented-domain value returned by the helper kernel."""
        return np.reshape(x.val, shape, order=order)

    kwargs['shape'] = shape
    kwargs['order'] = order 
    return _function_over_one_var(repr_func=_reshape_repr, raw_func=_reshape_raw, x=a, out=out, out_like=out_like, sizing=sizing, method=method, optimal_size=None, **kwargs)
