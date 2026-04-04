"""Array-shape, indexing-like, and linear-algebra-ish operators."""

import numpy as np

from ..objects import Fxp, implements
from .. import utils
from ..helpers import _cast_func, _use_object_cast
from .core import _function_over_one_var, _function_over_two_vars
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

        if val_min is not None:
            val_min = cast(val_min) * cast(2**x.n_frac)
        if val_max is not None:
            val_max = cast(val_max) * cast(2**x.n_frac)

        return cast(utils.clip(cast(x.val), val_min=val_min, val_max=val_max, **kwargs)) * cast(2**shift)

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
