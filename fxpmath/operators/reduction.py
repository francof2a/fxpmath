"""Reduction-oriented operators for fixed-point arrays."""

import numpy as np

from ..objects import Fxp, implements
from .. import utils
from ..helpers import _cast_func, _use_object_cast
from .core import _function_over_one_var
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

