import os
import sys
from packaging import version

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



import numpy as np
import pytest

import fxpmath as fxp
from fxpmath.objects import Fxp





def test_ufunc():
    """Validates ufunc by checking complex fixed-point behavior, NumPy interoperability."""
    vx = [-1., 0., 1.]
    vy = [1., 2., 4.]
    vc = [1j*0.5, 1.5 + 1j*2.0, -0.5 + 1j*0]


    nx = np.asarray(vx)
    ny = np.asarray(vy)
    nc = np.asarray(vc)

    fx = Fxp(vx, True, 16, 8)
    fy = Fxp(vy, True, 12, 4)
    fc = Fxp(vc, True, 12, 4)

    c = 2.

    ufunc_one_param_list = [
        np.positive,
        np.negative,
        np.conj,
        np.exp,
        np.abs,
        np.sin,
    ] 

    for ufunc in ufunc_one_param_list:
        assert (ufunc(nx) == ufunc(fx)()).all()
        assert (ufunc(ny) == ufunc(fy)()).all()
        assert (ufunc(nc) == ufunc(fc)()).all()

    ufunc_one_positive_param_list = [
        np.log,
        np.log10,
        np.sqrt,
    ] 

    for ufunc in ufunc_one_positive_param_list:
        assert np.allclose(ufunc(ny), ufunc(fy)(), rtol=fy.precision)

    ufunc_two_params_list = [
        np.add,
        np.subtract,
        np.multiply,
        np.divide,
    ]

    for ufunc in ufunc_two_params_list:
        assert (ufunc(nx, c) == ufunc(fx, c)()).all()
        assert (ufunc(ny, c) == ufunc(fy, c)()).all()
        assert (ufunc(nx, ny) == ufunc(fx, fy)()).all()
        assert (ufunc(nx, ny) == ufunc(nx, fy)()).all()
        assert (ufunc(nx, ny) == ufunc(fx, ny)()).all()

    ufunc_two_array_params_list = [
        np.matmul,
    ]

    for ufunc in ufunc_two_array_params_list:
        assert (ufunc(nx, ny) == ufunc(fx, fy)()).all()
        assert (ufunc(nx, ny) == ufunc(nx, fy)()).all()
        assert (ufunc(nx, ny) == ufunc(fx, ny)()).all()

def test_reduce_func():
    """Validates reduce func by checking complex fixed-point behavior, NumPy reduction interoperability, NumPy dot-product interoperability."""
    vx = [-1., 0., 1.]
    vy = [1., 2., 3.]
    vc = [1j*0.5, 1.5 + 1j*2.0, -0.5 + 1j*0]


    nx = np.asarray(vx)
    ny = np.asarray(vy)
    nc = np.asarray(vc)

    fx = Fxp(vx, True, 16, 8)
    fy = Fxp(vy, True, 12, 4)
    fc = Fxp(vc, True, 12, 4)

    c = 2.5

    ufunc_one_param_list = [
        np.sum,
        np.mean,
    ] 

    for ufunc in ufunc_one_param_list:
        assert (ufunc(nx) == ufunc(fx)()).all()
        assert (ufunc(ny) == ufunc(fy)()).all()
        assert (ufunc(nc) == ufunc(fc)()).all()

    ufunc_two_array_params_list = [
        np.inner,
        np.dot
    ]

    for ufunc in ufunc_two_array_params_list:
        assert (ufunc(nx, ny) == ufunc(fx, fy)()).all()
        assert (ufunc(nx, ny) == ufunc(nx, fy)()).all()
        assert (ufunc(nx, ny) == ufunc(fx, ny)()).all()

def test_ndarray_methods():
    """Validates ndarray methods by checking NumPy interoperability."""
    values = [[1, 2, 3], [-1, 0, 1]]
    w = Fxp(values, True, 16, 8)
    wa = np.array(values)

    func_list = [
        'all',
        'any',
        'max',
        'min',
        'mean',
        'sum',
        'cumsum',
        'cumprod',
        'prod',
    ]

    for func in func_list:
        assert (np.array(getattr(w, func)()) == np.array(getattr(wa, func)())).all()
        assert (np.array(getattr(w, func)(axis=0)) == getattr(wa, func)(axis=0)).all()
        assert (np.array(getattr(w, func)(axis=1)) == getattr(wa, func)(axis=1)).all()

    # close comparison
    func_list = [
        'var',
        'std'
    ]
    for func in func_list:
        assert np.allclose(np.array(getattr(w, func)()), np.array(getattr(wa, func)()), rtol=1/2**8)
        assert np.allclose(np.array(getattr(w, func)(axis=0)), np.array(getattr(wa, func)(axis=0)), rtol=1/2**8)
        assert np.allclose(np.array(getattr(w, func)(axis=1)), np.array(getattr(wa, func)(axis=1)), rtol=1/2**8)

    # no axis
    func_list = [
        'conjugate',
        'transpose',
        'diagonal',
        'trace',
    ]
    for func in func_list:
        assert (np.array(getattr(w, func)()) == np.array(getattr(wa, func)())).all()


    # in place
    func_list = [
        'sort',
    ]
    for func in func_list:
        getattr(w, func)()
        getattr(wa, func)()
        assert (np.array(w) == np.array(wa)).all()

    # return ndarray
    func_list = [
        'argmin',
        'argmax',
        'argsort',
        'nonzero',
    ]
    for func in func_list:
        r = getattr(w, func)()
        ra = getattr(wa, func)()

        if isinstance(r, tuple):
            for r_val, ra_val in zip(r, ra):
                assert (r_val == ra_val).all()   
        else:
            assert (r == ra).all()   

def test_outputs_formats():
    
    """Validates outputs formats by checking NumPy interoperability."""
    values = [[1, 2, 3], [-1, 0, 1]]
    w = Fxp(values, True, 16, 8)
    like_ref = Fxp(None, True, 24, 12)
    out_ref = Fxp(None, True, 24, 8)

    if version.parse(np.__version__) >= version.parse('1.21'):
        # since numpy 1.21 unknown arguments raise an error
        # by now only test the fxpmath.functions.add instead of numpy dispatched add function
        from fxpmath.functions import add
        z = add(w, 2, out_like=like_ref)
    else:
        z = np.add(w, 2, out_like=like_ref)

    assert isinstance(z, Fxp)
    assert z.n_frac == like_ref.n_frac
    assert z.n_int == like_ref.n_int
    assert (z.get_val() == np.array(values) + 2).all()

    z = np.add(w, 2, out=out_ref)

    assert isinstance(z, Fxp)
    assert z is out_ref
    assert z.n_frac == out_ref.n_frac
    assert z.n_int == out_ref.n_int
    assert (z.get_val() == np.array(values) + 2).all()

    # np.std(w, out_like=like_ref)

def test_numpy_out_ellipsis_add_dispatch():
    """Validates NumPy add dispatch by checking out=... compatibility."""
    x = Fxp([1, 2, 3], True, 16, 0)
    y = np.add(x, 2, out=...)

    assert isinstance(y, Fxp)
    assert np.all(y() == np.array([3, 4, 5]))


def test_numpy_out_ellipsis_sum_dispatch():
    """Validates NumPy sum dispatch by checking out=... compatibility."""
    x = Fxp([1, 2, 3], True, 16, 0)
    y = np.sum(x, out=...)

    assert isinstance(y, Fxp)
    assert y() == 6


def test_numpy_out_ellipsis_prod_dispatch():
    """Validates NumPy prod dispatch by checking out=... compatibility."""
    x = Fxp([1, 2, 3], True, 16, 0)
    y = np.prod(x, out=...)

    assert isinstance(y, Fxp)
    assert y() == 6


def test_numpy_out_ellipsis_dot_dispatch():
    """Validates NumPy dot dispatch by checking out=... compatibility."""
    x = Fxp([1, 2, 3], True, 16, 0)
    z = Fxp([1, 1, 1], True, 16, 0)
    y = np.dot(x, z, out=...)

    assert isinstance(y, Fxp)
    assert y() == 6


def test_numpy_out_ellipsis_sum_scalar_dispatch():
    """Validates scalar NumPy sum dispatch by checking out=... compatibility."""
    x = Fxp(5, True, 16, 0)
    y = np.sum(x, out=...)

    assert isinstance(y, Fxp)
    assert y() == 5


def test_numpy_out_ellipsis_sum_zerod_dispatch():
    """Validates zero-dimensional NumPy sum dispatch by checking out=... compatibility."""
    x = Fxp(np.array(5), True, 16, 0)
    y = np.sum(x, out=...)

    assert isinstance(y, Fxp)
    assert y() == 5


def test_numpy_out_fxp_reuse_dot_dispatch():
    """Validates NumPy dot dispatch by checking explicit Fxp out reuse."""
    x = Fxp([1, 2, 3], True, 16, 0)
    z = Fxp([1, 1, 1], True, 16, 0)
    out_ref = Fxp(None, True, 32, 0)
    y = np.dot(x, z, out=out_ref)

    assert isinstance(y, Fxp)
    assert y is out_ref
    assert y() == 6


def test_numpy_out_invalid_type_raises_typeerror_add_dispatch():
    """Validates NumPy add dispatch by checking invalid out type raises TypeError."""
    x = Fxp([1, 2, 3], True, 16, 0)

    with pytest.raises(TypeError, match='`out` must be a Fxp object!'):
        np.add(x, 2, out=np.empty(3, dtype=float))


def test_numpy_out_invalid_type_raises_typeerror_sum_dispatch():
    """Validates NumPy sum dispatch by checking invalid out type raises TypeError."""
    x = Fxp([1, 2, 3], True, 16, 0)

    with pytest.raises(TypeError, match='`out` must be a Fxp object!'):
        np.sum(x, out=np.empty((), dtype=float))


def test_numpy_array_protocol_array_copy_dtype_behavior():
    """Validates NumPy array protocol by checking dtype/copy behavior."""
    x = Fxp([1, 2, 3], True, 16, 0)

    arr_dtype = np.asarray(x, dtype=np.float64)
    assert arr_dtype.dtype == np.float64
    assert np.all(arr_dtype == np.array([1.0, 2.0, 3.0]))

    arr_no_copy = np.array(x, copy=False)
    assert np.all(arr_no_copy == np.array([1, 2, 3]))

    if version.parse(np.__version__) >= version.parse('2.0'):
        with pytest.raises(ValueError, match='Unable to avoid copy while creating an array as requested.'):
            np.array(x, dtype=np.float64, copy=False)
    else:
        arr_legacy = np.array(x, dtype=np.float64, copy=False)
        assert arr_legacy.dtype == np.float64
        assert np.all(arr_legacy == np.array([1.0, 2.0, 3.0]))


def test_numpy_array_protocol_legacy_array_args_backward_compatibility():
    """Validates NumPy array protocol by checking legacy positional arg compatibility."""
    x = Fxp([1, 2, 3], True, 16, 0)

    arr_legacy = x.__array__(np.float64)
    assert arr_legacy.dtype == np.float64
    assert np.all(arr_legacy == np.array([1.0, 2.0, 3.0]))
