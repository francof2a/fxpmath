"""Core operator helpers and lightweight public constructors."""

import numpy as np

from ..objects import Fxp
from .. import utils
from ..helpers import _cast_func, _use_object_cast
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
