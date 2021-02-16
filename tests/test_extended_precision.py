import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import numpy as np

import fxpmath as fxp
from fxpmath.objects import Fxp
from fxpmath import utils

import decimal
from decimal import Decimal
import math



def test_creation():
    x = Fxp(1.0, True, 256, 248)
    y = Fxp(-0.1, True, 128, 96)

    assert x() == 1.0
    assert y() == -0.1

    decimal.getcontext().prec = int(np.ceil(math.log10(2**248)))

    h_decimal = Decimal(1) / Decimal(3)
    h = Fxp(h_decimal, True, 256, 248)

    assert h.astype(float) == float(h_decimal)

    h = Fxp(h_decimal)
    assert h.astype(float) == float(h_decimal)

    w_vals = [0.1, 0.2, 0.3, 0.4, 0.5]
    w = Fxp(w_vals, True, 256, 248)
    assert (w() == np.array(w_vals)).all()

def test_math_operators():
    x = Fxp(1.0, True, 256, 248)
    y = Fxp(-0.1, True, 128, 96)

    z = x + y
    assert z() == 0.9

    assert (x-y)() == 1.1
    assert (x*y)() == -0.1
    assert (x/y)() == -10.0

def test_operations_with_combinations():
    
    v = [-256, -64, -16, -4.75, -3.75, -3.25, -1, -0.75, -0.125, 0.0, 0.125, 0.75, 1, 1.5, 3.75, 4.0, 8.0, 32, 128]
    for i in range(len(v)):
        for j in range(len(v)):
            vx, vy = v[i], v[j]
            x = Fxp(vx, True, 256, 240)
            y = Fxp(vy, True, 128, 96)
            assert (vx + vy) == (x + y)()
            assert (vy + vx) == (y + x)()

            assert (vx - vy) == (x - y)()
            assert -(vy - vx) == -(y - x)()

            assert (vx * vy) == (x * y)()
            assert (vy * vx) == (y * x)()

    v = [-256, -64, -16, -4.75, -4.25, -1, -0.75, -0.125, 0.125, 0.75, 1, 1.5, 2.75, 4.0, 8.0, 32, 128]
    d = [-256, -64, -16, -1, -0.5, -0.125, 0.125, 0.5, 1, 2, 4.0, 8.0, 32, 128]
    for i in range(len(v)):
        for j in range(len(d)):
            vx, vy = v[i], d[j]
            x = Fxp(vx, True, 256, 240)
            y = Fxp(vy, True, 128, 96)

            assert (vx / vy) == (x / y)()

            assert (vx // vy) == (x // y)()

            assert (vx % vy) == (x % y)()

def test_operations_with_constants_with_combinations():
    
    v = [-256, -64, -16, -4.75, -3.75, -3.25, -1, -0.75, -0.125, 0.0, 0.125, 0.75, 1, 1.5, 3.75, 4.0, 8.0, 32, 128]
    for i in range(len(v)):
        for j in range(len(v)):
            vx, vy = v[i], v[j]
            x = Fxp(vx, True, 256, 224)
            y = Fxp(vy, True, 128, 96)
            assert (x + vy)() == (vx + vy) == (vx + y)() == (x + y)()
            assert (vy + x)() == (vy + vx) == (y + vx)() == (y + x)()

            assert (x - vy)() == (vx - vy) == (vx - y)() == (x - y)()
            assert -(vy - x)() == -(vy - vx) == -(y - vx)() == -(y - x)()

    for i in range(len(v)):
        for j in range(len(v)):
            vx, vy = v[i], v[j]
            x = Fxp(vx, True, 256, 224)
            y = Fxp(vy, True, 128, 96)

            assert (x * vy)() == (vx * vy) == (vx * y)() == (x * y)()
            assert (vy * x)() == (vy * vx) == (y * vx)() == (y * x)()

    v = [-256, -64, -16, -4.75, -4.25, -1, -0.75, -0.125, 0.125, 0.75, 1, 1.5, 2.75, 4.0, 8.0, 32, 128]
    d = [-256, -64, -16, -1, -0.5, -0.125, 0.125, 0.5, 1, 2, 4.0, 8.0, 32, 128]
    for i in range(len(v)):
        for j in range(len(d)):
            vx, vy = v[i], d[j]
            x = Fxp(vx, True, 32, 12)
            y = Fxp(vy, True, 32, 12)

            assert (x / vy)() == (vx / vy) == (vx / y)() == (x / y)()
            # assert (vy / x)() == (vy / vx) == (y / vx)() == (y / x)()

            assert (x // vy)() == (vx // vy) == (vx // y)() == (x // y)()
            # assert (vy // x)() == (vy // vx) == (y // vx)() == (y // x)()

            assert (x % vy)() == (vx % vy) == (vx % y)() == (x % y)()
            # assert (vy % x)() == (vy % vx) == (y % vx)() == (y % x)()

def test_numpy_ufunc():
    vx = [-1., 0., 1.]
    vy = [1., 2., 4.]
    vc = [1j*0.5, 1.5 + 1j*2.0, -0.5 + 1j*0]


    nx = np.asarray(vx)
    ny = np.asarray(vy)
    nc = np.asarray(vc)

    fx = Fxp(vx, True, 16*8, 8*8)
    fy = Fxp(vy, True, 12*8, 4*8)
    fc = Fxp(vc, True, 12*8, 4*8)

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