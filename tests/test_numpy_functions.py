import os
import sys
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