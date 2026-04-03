"""Binary arithmetic operator kernels and public arithmetic API."""

import numpy as np

from ..objects import Fxp, implements
from .. import utils
from ..helpers import _cast_func, _use_object_cast
from .core import _function_over_two_vars

try:
    from decimal import Decimal
except:
    Decimal = type(None)
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
        # Unsigned subtraction requires signed-capable intermediates; native unsigned
        # array arithmetic can wrap underflow before overflow handling is applied.
        if not x.signed and not y.signed:
            use_object = True
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

