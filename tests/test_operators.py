import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import fxpmath as fxp
from fxpmath.objects import Fxp
from fxpmath import utils

import numpy as np

def test_shift_bitwise():
    # integer val
    x = Fxp(32, True, 8, 0)
    # left
    assert (x << 1)() == 64
    assert (x << 2)() == 128
    assert (x << 2).n_word == 9 
    assert (x << 3)() == 256
    assert (x << 10)() == 32*(2**10)
    # right
    assert (x >> 1)() == 16
    assert (x >> 2)() == 8
    assert (x >> 3)() == 4
    assert (x >> 5)() == 1
    assert (x >> 6)() == 0.5

    # float val
    x = Fxp(24.25, True, 8, 2)
    #left
    assert (x << 1)() == 48.5
    assert (x << 4)() == 388.0
    #right
    x = Fxp(24.5, True, 8, 2)
    assert (x >> 1)() == 12.25
    assert (x >> 2)() == 6.125

    # negative
    x = Fxp(-24.25, True, 8, 2)
    #left
    assert (x << 1)() == -48.5
    assert (x << 4)() == -388.0
    #right
    x = Fxp(-24.5, True, 8, 2)
    assert (x >> 1)() == -12.25
    assert (x >> 2)() == -6.125

    # trunc shift
    # left
    x = Fxp(32, True, 8, 0, shifting='trunc')
    assert (x << 1)() == 64
    assert (x << 2)() == x.upper
    # right
    assert (x >> 3)() == 4
    assert (x >> 5)() == 1
    assert (x >> 6)() == 0    

    # unsigned

    x = Fxp(32, False, 8, 0)
    # left
    assert (x << 1)() == 64
    assert (x << 2)() == 128 
    assert (x << 3)() == 256
    assert (x << 3).n_word == 9
    assert (x << 10)() == 32*(2**10)
    # right
    assert (x >> 1)() == 16
    assert (x >> 2)() == 8
    assert (x >> 3)() == 4
    assert (x >> 5)() == 1
    assert (x >> 6)() == 0.5

    # float val
    x = Fxp(24.25, False, 8, 2)
    #left
    assert (x << 1)() == 48.5
    assert (x << 4)() == 388.0
    #right
    x = Fxp(24.5, False, 8, 2)
    assert (x >> 1)() == 12.25
    assert (x >> 2)() == 6.125

    # trunc left shift
    x = Fxp(64, False, 8, 0, shifting='trunc')
    assert (x << 1)() == 128
    assert (x << 2)() == x.upper

def test_invert():
    x = Fxp(None, True, 8, 4)
    xu = Fxp(None, False, 8, 4)

    x('0b 0010 1100')
    y = ~x
    assert y.bin() == '11010011'

    x('0b0000 0000')
    assert (~x).bin() == '11111111'
    xu('0b0000 0000')
    assert (~xu).bin() == '11111111'

    x('0b 1111 1111')
    assert (~x).bin() == '00000000'
    xu('0b 1111 1111')
    assert (~xu).bin() == '00000000'

    x('0b 1000 0000')
    assert (~x).bin() == '01111111'
    xu('0b 1000 0000')
    assert (~xu).bin() == '01111111'

    x = Fxp(None, True, 32, 0)
    xu = Fxp(None, False, 32, 0)

    val_str = '10100000111101011100001100110101'
    inv_str = '01011111000010100011110011001010'
    x('0b'+val_str)
    assert (~x).bin() == inv_str
    xu('0b'+val_str)
    assert (~xu).bin() == inv_str

def test_and():
    x = Fxp(None, True, 8, 4)
    xu = Fxp(None, False, 8, 4)
    y = Fxp(None, True, 8, 4)
    yu = Fxp(None, False, 8, 4)

    val_str = '00110101'
    mks_str = '11110000'
    and_str = '00110000'

    x('0b'+val_str)
    xu('0b'+val_str)
    y('0b'+mks_str)
    yu('0b'+mks_str)

    assert (x & y).bin() == and_str
    assert (x & yu).bin() == and_str
    assert (xu & y).bin() == and_str
    assert (xu & yu).bin() == and_str
    assert (x & utils.str2num('0b'+mks_str)).bin() == and_str
    assert (xu & utils.str2num('0b'+mks_str)).bin() == and_str
    assert (utils.str2num('0b'+mks_str) & x).bin() == and_str
    assert (utils.str2num('0b'+mks_str) & xu).bin() == and_str

    val_str = '10101100'
    mks_str = '11001100'
    and_str = '10001100'

    x('0b'+val_str)
    xu('0b'+val_str)
    y('0b'+mks_str)
    yu('0b'+mks_str)

    assert (x & y).bin() == and_str
    assert (x & yu).bin() == and_str
    assert (xu & y).bin() == and_str
    assert (xu & yu).bin() == and_str
    assert (x & utils.str2num('0b'+mks_str)).bin() == and_str
    assert (xu & utils.str2num('0b'+mks_str)).bin() == and_str
    assert (utils.str2num('0b'+mks_str) & x).bin() == and_str
    assert (utils.str2num('0b'+mks_str) & xu).bin() == and_str

def test_or():
    x = Fxp(None, True, 8, 4)
    xu = Fxp(None, False, 8, 4)
    y = Fxp(None, True, 8, 4)
    yu = Fxp(None, False, 8, 4)

    val_str = '00110101'
    mks_str = '11110000'
    or_str  = '11110101'

    x('0b'+val_str)
    xu('0b'+val_str)
    y('0b'+mks_str)
    yu('0b'+mks_str)

    assert (x | y).bin() == or_str
    assert (x | yu).bin() == or_str
    assert (xu | y).bin() == or_str
    assert (xu | yu).bin() == or_str
    assert (x | utils.str2num('0b'+mks_str)).bin() == or_str
    assert (xu | utils.str2num('0b'+mks_str)).bin() == or_str
    assert (utils.str2num('0b'+mks_str) | x).bin() == or_str
    assert (utils.str2num('0b'+mks_str) | xu).bin() == or_str

    val_str = '10101100'
    mks_str = '11001100'
    or_str  = '11101100'

    x('0b'+val_str)
    xu('0b'+val_str)
    y('0b'+mks_str)
    yu('0b'+mks_str)

    assert (x | y).bin() == or_str
    assert (x | yu).bin() == or_str
    assert (xu | y).bin() == or_str
    assert (xu | yu).bin() == or_str
    assert (x | utils.str2num('0b'+mks_str)).bin() == or_str
    assert (xu | utils.str2num('0b'+mks_str)).bin() == or_str
    assert (utils.str2num('0b'+mks_str) | x).bin() == or_str
    assert (utils.str2num('0b'+mks_str) | xu).bin() == or_str

def test_xor():
    x = Fxp(None, True, 8, 4)
    xu = Fxp(None, False, 8, 4)
    y = Fxp(None, True, 8, 4)
    yu = Fxp(None, False, 8, 4)

    val_str = '00110101'
    mks_str = '11110000'
    xor_str = '11000101'

    x('0b'+val_str)
    xu('0b'+val_str)
    y('0b'+mks_str)
    yu('0b'+mks_str)

    assert (x ^ y).bin() == xor_str
    assert (x ^ yu).bin() == xor_str
    assert (xu ^ y).bin() == xor_str
    assert (xu ^ yu).bin() == xor_str
    assert (x ^ utils.str2num('0b'+mks_str)).bin() == xor_str
    assert (xu ^ utils.str2num('0b'+mks_str)).bin() == xor_str
    assert (utils.str2num('0b'+mks_str) ^ x).bin() == xor_str
    assert (utils.str2num('0b'+mks_str) ^ xu).bin() == xor_str

    val_str = '10101100'
    mks_str = '11001100'
    xor_str = '01100000'

    x('0b'+val_str)
    xu('0b'+val_str)
    y('0b'+mks_str)
    yu('0b'+mks_str)

    assert (x ^ y).bin() == xor_str
    assert (x ^ yu).bin() == xor_str
    assert (xu ^ y).bin() == xor_str
    assert (xu ^ yu).bin() == xor_str
    assert (x ^ utils.str2num('0b'+mks_str)).bin() == xor_str
    assert (xu ^ utils.str2num('0b'+mks_str)).bin() == xor_str
    assert (utils.str2num('0b'+mks_str) ^ x).bin() == xor_str
    assert (utils.str2num('0b'+mks_str) ^ xu).bin() == xor_str

def test_arrays():
    x = Fxp(None, True, 8, 4)
    y = Fxp(None, True, 8, 4)

    x(['0b00110101', '0b10101100'])
    y('0b11110000')

    z = x & y
    assert z.bin()[0] == '00110000'
    assert z.bin()[1] == '10100000'

def test_operations_with_combinations():
    
    v = [-256, -64, -16, -4.75, -3.75, -3.25, -1, -0.75, -0.125, 0.0, 0.125, 0.75, 1, 1.5, 3.75, 4.0, 8.0, 32, 128]
    for i in range(len(v)):
        for j in range(len(v)):
            vx, vy = v[i], v[j]
            x = Fxp(vx)
            y = Fxp(vy)
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
            x = Fxp(vx)
            y = Fxp(vy)

            assert (vx / vy) == (x / y)()

            assert (vx // vy) == (x // y)()

            assert (vx % vy) == (x % y)()

def test_operations_with_constants_with_combinations():
    
    v = [-256, -64, -16, -4.75, -3.75, -3.25, -1, -0.75, -0.125, 0.0, 0.125, 0.75, 1, 1.5, 3.75, 4.0, 8.0, 32, 128]
    for i in range(len(v)):
        for j in range(len(v)):
            vx, vy = v[i], v[j]
            x = Fxp(vx, True, 16, 3)
            y = Fxp(vy, True, 16, 3)
            assert (x + vy)() == (vx + vy) == (vx + y)() == (x + y)()
            assert (vy + x)() == (vy + vx) == (y + vx)() == (y + x)()

            assert (x - vy)() == (vx - vy) == (vx - y)() == (x - y)()
            assert -(vy - x)() == -(vy - vx) == -(y - vx)() == -(y - x)()

    for i in range(len(v)):
        for j in range(len(v)):
            vx, vy = v[i], v[j]
            x = Fxp(vx, True, 24, 6)
            y = Fxp(vy, True, 24, 6)

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

def test_pow():
    x = Fxp(16, True, n_int=14, n_frac=8)
    n = Fxp(-1, True, n_int=14, n_frac=8)
    assert(x**n)() == 1/16

    v = 15
    n_vals = [0, 1, 2, 3]

    x = Fxp(v, signed=True, n_int=12, n_frac=0)
    xu = Fxp(v, signed=False, n_int=12, n_frac=0)
    for n in n_vals:
        assert (x**n)() == v**n
        assert (xu**n)() == v**n
    
    v = -16
    x = Fxp(v, signed=True, n_int=12, n_frac=0)
    for n in n_vals:
        assert (x**n)() == v**n

    v = 16.0
    n_vals = [-2, -1, 0, 1, 2, 3]

    x = Fxp(v, signed=True, n_int=14, n_frac=8)
    # xu = Fxp(v, signed=False, n_int=12, n_frac=0)
    for n in n_vals:
        assert (x**n)() == v**n
        # assert (xu**n)() == v**n
    
    v = -16.0
    x = Fxp(v, signed=True, n_int=14, n_frac=8)
    for n in n_vals:
        assert (x**n)() == (v)**n

    v = 81
    n_vals = [0, 0.25, 0.5]

    x = Fxp(v, signed=True, n_int=14, n_frac=8)
    xu = Fxp(v, signed=False, n_int=14, n_frac=8)
    for n in n_vals:
        assert (x**n)() == v**n
        assert (xu**n)() == v**n


    v = 16.
    n = 2
    v_vals = [-4, -2, -1, 0, 1, 2, 4]
    n_vals = [-2, -1, 0, 1, 2]

    x = Fxp(v, signed=True, n_int=12, n_frac=0)
    xu = Fxp(v, signed=False, n_int=12, n_frac=0)
    p = Fxp(n, signed=True, n_int=8, n_frac=0)

    assert ((x**p)() == np.power(v, n)).all()
    assert ((xu**p)() == np.power(v, n)).all()

    x = Fxp(v, signed=True, n_int=12, n_frac=8)
    p_vals = Fxp(n_vals, signed=True, n_int=8, n_frac=0)
    x.config.op_sizing = 'same'
    assert ((x**p_vals)() == np.power(v, n_vals)).all()
    p_vals = Fxp(n_vals, signed=True, n_int=8, n_frac=0)
    assert ((x**p_vals)() == np.power(v, n_vals)).all()

    x_vals = Fxp(v_vals, signed=True, n_int=12, n_frac=8)
    p = Fxp(n, signed=True, n_int=8, n_frac=0)
    x_vals.config.op_sizing = 'same'
    assert ((x_vals**p)() == np.power(v_vals, n)).all()
    p = Fxp(n, signed=True, n_int=8, n_frac=2)
    assert ((x_vals**p)() == np.power(v_vals, n)).all()

    v_vals = [-1, 1, 2, 3, 4]
    n_vals = [-2, -1, 0, 1, 2]
    x_vals = Fxp(v_vals, signed=True, n_int=12, n_frac=8)
    p_vals = Fxp(n_vals, signed=True, n_int=8, n_frac=0)
    x_vals.config.op_sizing = 'same'
    assert ((x_vals**p_vals)() == np.array([vi**ni for vi, ni in zip(v_vals, n_vals)])).all()
    p_vals = Fxp(n_vals, signed=True, n_int=8, n_frac=2)
    assert ((x_vals**p_vals)() == np.array([vi**ni for vi, ni in zip(v_vals, n_vals)])).all()

    v_vals = [[1, 2],[3, 4]]
    n_vals = [[1, 2],[3, 4]]
    x_vals = Fxp(v_vals, signed=True, n_int=12, n_frac=8)
    p_vals = Fxp(n_vals, signed=True, n_int=8, n_frac=0)
    x_vals.config.op_sizing = 'same'
    assert ((x_vals**p_vals)() == np.power(v_vals, n_vals)).all()

def test_scaled():
    x = Fxp(10.5, True, 16, 8, scale=2, bias=1)

    assert x() == 10.5
    
    assert x + 2 == 12.5
    assert x - 2.5 == 8.0
    assert x * 3 == 31.5
    assert x / 2 == 5.25

def test_abs():
    x = Fxp(-3.5, True, 32, 16)

    assert x() == -3.5
    assert abs(x)() == 3.5

    x = Fxp(3.5, True, 32, 16)
    assert abs(x)() == 3.5
    