import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import fxpmath as fxp
from fxpmath.objects import Fxp
import fxpmath.utils as fxp_utils

import warnings

import numpy as np

def test_complex_creation():
    """Validates complex creation by checking complex fixed-point behavior, dtype parsing and conversion behavior."""
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
    """Validates math operations by checking complex fixed-point behavior, NumPy interoperability, dtype parsing and conversion behavior."""
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
    """Validates complex repr by checking hexadecimal parsing/formatting paths, complex fixed-point behavior, NumPy interoperability."""
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



def test_complex_bitwise_diverse_scalar_array_mixed():
    """Validates diverse scalar/array, mixed, and complex bitwise operations across multiple sizes."""
    fxp_utils.reset_mixed_complex_bitwise_warning_state()

    x = Fxp(0b11+0b11*1j, dtype='fxp-u2/0-complex')

    # Mixed complex/non-complex: warning once, but operations remain component-wise.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        z_and = x & 0b01
        z_or = x | 0b10
        z_xor = x ^ 0b11

    assert z_and() == (1+1j)
    assert z_or() == (3+3j)
    assert z_xor() == 0j
    mixed_msgs = [w for w in caught if issubclass(w.category, fxp_utils.ComplexBitwiseOperationWarning)]
    assert len(mixed_msgs) == 1

    # Complex-complex scalar ops are part-wise.
    y = Fxp(0b01+0b10*1j, dtype='fxp-u2/0-complex')
    assert (x & y)() == (1+2j)
    assert (x | y)() == (3+3j)
    assert (x ^ y)() == (2+1j)

    # Reverse mixed path (real Fxp with complex scalar) remains part-wise and emits no extra warning.
    with warnings.catch_warnings(record=True) as caught_again:
        warnings.simplefilter("always")
        real = Fxp(0b10, dtype='fxp-u2/0')
        rev_and = real & (0b01+0b10*1j)
        rev_or = real | (0b01+0b10*1j)
        rev_xor = real ^ (0b01+0b10*1j)

    assert rev_and() == (0+2j)
    assert rev_or() == (3+2j)
    assert rev_xor() == (3+0j)
    assert len(caught_again) == 0

    # Unary invert applies to both components.
    inv = ~Fxp(0b01+0b10*1j, dtype='fxp-u2/0-complex')
    assert inv() == (2+1j)

    # Array values: element-wise ops against a complex scalar operand.
    a = Fxp(np.array([0b11+0b01*1j, 0b01+0b10*1j]), dtype='fxp-u2/0-complex')
    c = 0b01 + 0b11*1j

    arr_and = a & c
    arr_or = a | c
    arr_xor = a ^ c
    arr_inv = ~a

    assert np.all(arr_and() == np.array([1+1j, 1+2j]))
    assert np.all(arr_or() == np.array([3+3j, 1+3j]))
    assert np.all(arr_xor() == np.array([2+2j, 0+1j]))
    assert np.all(arr_inv() == np.array([0+2j, 2+1j]))

    b = Fxp(np.array([0b01+0b11*1j, 0b10+0b01*1j]), dtype='fxp-u2/0-complex')
    arr_arr_and = a & b
    arr_arr_or = a | b
    arr_arr_xor = a ^ b

    assert np.all(arr_arr_and() == np.array([1+1j, 0+0j]))
    assert np.all(arr_arr_or() == np.array([3+3j, 3+3j]))
    assert np.all(arr_arr_xor() == np.array([2+2j, 3+3j]))

    # Size diversity: unsigned fractional complex (u4/1).
    u = Fxp(1.5 + 0.5j, dtype='fxp-u4/1-complex')
    v = Fxp(0.5 + 1.5j, dtype='fxp-u4/1-complex')

    assert (u & v)() == (0.5+0.5j)
    assert (u | v)() == (1.5+1.5j)
    assert (u ^ v)() == (1+1j)
    assert (~u)() == (6+7j)

    # Size diversity: signed fractional complex (s5/1).
    us = Fxp(-1.5 + 0.5j, dtype='fxp-s5/1-complex')
    vs = Fxp(0.5 - 1.5j, dtype='fxp-s5/1-complex')

    assert (us & vs)() == (0.5+0.5j)
    assert (us | vs)() == (-1.5-1.5j)
    assert (us ^ vs)() == (-2-2j)
    assert (~us)() == (1-1j)

    # Mixed scalar with fractional formats still applies to both parts.
    assert (u & 0b01)() == (0.5+0.5j)
    assert (u | 0b10)() == (1.5+1.5j)
    assert (u ^ 0b11)() == 1j
    assert (us & 0b01)() == (0.5+0.5j)
    assert (us | 0b10)() == (-0.5+1.5j)
    assert (us ^ 0b11)() == (-1+1j)
