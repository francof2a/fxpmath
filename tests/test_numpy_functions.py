import os
import sys
from packaging import version

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



import numpy as np

import fxpmath as fxp
from fxpmath.objects import Fxp





def test_ufunc():
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