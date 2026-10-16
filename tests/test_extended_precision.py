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
    """Verify creation of high-precision fixed-point values from floats, decimals, and arrays."""
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
    """Verify core arithmetic operators preserve expected results for extended-precision operands."""
    x = Fxp(1.0, True, 256, 248)
    y = Fxp(-0.1, True, 128, 96)

    z = x + y
    assert z() == 0.9

    assert (x-y)() == 1.1
    assert (x*y)() == -0.1
    assert (x/y)() == -10.0

def test_operations_with_combinations():
    """Exhaustively compare extended-precision Fxp-vs-Fxp arithmetic against Python numeric results."""
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
    """Exhaustively compare extended-precision mixed Fxp/constant arithmetic against Python numeric results."""
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
    """Validates numpy ufunc by checking complex fixed-point behavior, NumPy interoperability."""
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


def test_astype_decimal():
    """Verify `astype(Decimal)` returns exact values for scalars, arrays, indexing, and scaled Fxp."""
    # Round trip from Decimal input beyond float64 precision.
    d = Decimal('27670116110564327421.25')
    x = Fxp(d, signed=True, n_int=65, n_frac=2)
    assert x.astype(Decimal) == d
    assert x.get_val(Decimal) == d
    assert x() == float(d)  # default output type is unchanged

    # Arrays return object arrays of Decimal; index/item return scalars.
    y = Fxp(None, True, 80, 2)
    y.set_val(np.array([2**74 + 1, -6, 5], dtype=object), raw=True)
    y_dec = y.astype(Decimal)
    assert y_dec.dtype == object
    assert list(y_dec) == [Decimal(2**72) + Decimal('0.25'), Decimal('-1.5'), Decimal('1.25')]
    assert y.astype(Decimal, index=0) == y_dec[0]
    assert y.astype(Decimal, item=1) == Decimal('-1.5')
    assert y[0].astype(Decimal) == y_dec[0]

    # Non-extended and negative n_frac Fxp.
    assert Fxp(-1.25, dtype='S16.8').astype(Decimal) == Decimal('-1.25')
    assert Fxp(96, dtype='S16.-3').astype(Decimal) == Decimal(96)

    # Scaling is applied exactly.
    z = Fxp(2**70 + 1, dtype='S80.4', scale=0.5, bias=1)
    assert z.astype(Decimal) == Decimal(2**70 + 1)

    # Global decimal context is not modified.
    prec = decimal.getcontext().prec
    Fxp(1.0, True, 256, 248).astype(Decimal)
    assert decimal.getcontext().prec == prec

    # Complex values are not supported.
    try:
        Fxp(1 + 1j, dtype='S80.4').astype(Decimal)
        assert False, 'TypeError expected'
    except TypeError:
        pass
