import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import fxpmath as fxp
from fxpmath.objects import Fxp

import numpy as np

def test_complex_creation():
    x = Fxp(0.25 - 1j*14.5)
    assert x() == 0.25 - 1j*14.5
    assert x.real == 0.25
    assert x.imag == -14.5
    assert x.dtype == 'fxp-s7/2-complex'
    assert x.vdtype == complex

    x = Fxp(3.0, dtype='fxp-s8/4-complex')
    assert x() == 3.0
    assert x.imag == 0.0

    x = Fxp(1j*3.0, dtype='fxp-s8/4-complex')
    assert x() == 1j*3.0
    assert x.real == 0.0
    assert x.imag == 3.0

    x = Fxp([0.0, 1.0 + 1j*1.0, -1j*2.5], signed=True, n_word=8)
    assert x.dtype == 'fxp-s8/1-complex'
    assert x[0]() == 0.0
    assert x[1]() == 1.0 + 1j*1.0
    assert x[2]() == -1j*2.5

    x = Fxp(0.25 - 1j*14.5, dtype='Q6.4')
    assert x.dtype == 'fxp-s10/4-complex'

def test_math_operations():
    c = 2.0
    x = 0.25 - 1j*14.5
    y = -1.0 + 1j*0.5

    x_fxp = Fxp(x, dtype='Q14.3')
    y_fxp = Fxp(y, dtype='Q14.3')

    # add
    z = x + y
    z_fxp = x_fxp + y_fxp
    assert z_fxp() == z

    z = x + c
    z_fxp = x_fxp + c
    assert z_fxp() == z

    # sub
    z = x - y
    z_fxp = x_fxp - y_fxp
    assert z_fxp() == z

    z = x - c
    z_fxp = x_fxp - c
    assert z_fxp() == z

    # mul
    z = x * y
    z_fxp = x_fxp * y_fxp
    assert z_fxp() == z

    z = x * c
    z_fxp = x_fxp * c
    assert z_fxp() == z    

    # div
    z = x / y
    z_fxp = x_fxp / y_fxp
    assert z_fxp() == z

    z = x / c
    z_fxp = x_fxp / c
    assert z_fxp() == z

    # floor div
    x = np.asarray(x)
    y = np.asarray(y)
    z = (x * y.conj()).real // (y * y.conj()).real + 1j* ((x * y.conj()).imag // (y * y.conj()).real)
    z_fxp = x_fxp // y_fxp
    assert z_fxp() == z

    c = np.asarray(c)
    z = (x * c.conj()).real // (c * c.conj()).real + 1j* ((x * c.conj()).imag // (c * c.conj()).real)
    z_fxp = x_fxp // c
    assert z_fxp() == z

    # abs
    x = -3.0 + 1j*4.0
    x_fxp = Fxp(x, dtype='Q16.16')

    assert abs(x_fxp)() == 5.0

def test_complex_repr():
    c_fxp = Fxp(1 + 1j*15)
    assert c_fxp.bin() == '00001+01111j'
    assert c_fxp.hex() == '0x01+0x0Fj'
    assert c_fxp.base_repr(base=2) == '1+1111j'
    assert c_fxp.base_repr(base=10) == '1+15j'
    assert c_fxp.base_repr(base=16) == '1+Fj'

    c_fxp = Fxp(3.5 - 1j*0.25)
    assert c_fxp.bin() == '01110+11111j'
    assert c_fxp.bin(frac_dot=True) == '011.10+111.11j'
    assert c_fxp.hex() == '0x0E+0x1Fj'
    assert c_fxp.base_repr(base=2) == '1110-1j'
    assert c_fxp.base_repr(base=2, frac_dot=True) == '11.10-.01j'
    assert c_fxp.base_repr(base=10) == '14-1j'
    assert c_fxp.base_repr(base=16) == 'E-1j'

    c_fxp = Fxp(12 - 1j*1)
    assert c_fxp.bin() == '01100+11111j'
    assert c_fxp.hex() == '0x0C+0x1Fj'
    assert c_fxp.base_repr(base=2) == '1100-1j'
    assert c_fxp.base_repr(base=10) == '12-1j'
    assert c_fxp.base_repr(base=16) == 'C-1j'

    arr_fxp = Fxp(np.array([[1 + 1j*2, 2 - 1j*3]]))
    assert np.all(arr_fxp.bin() == np.array(['001+010j', '010+101j']))
    assert np.all(arr_fxp.hex() == np.array(['0x1+0x2j', '0x2+0x5j']))
    assert np.all(arr_fxp.base_repr(base=2) == np.array(['1+10j', '10-11j']))
    assert np.all(arr_fxp.base_repr(base=10) == np.array(['1+2j', '2-3j']))
    assert np.all(arr_fxp.base_repr(base=16) == np.array(['1+2j', '2-3j']))

