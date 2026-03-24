import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import fxpmath as fxp
from fxpmath.objects import Fxp
from fxpmath import utils

import numpy as np

def test_shift_bitwise():
    # integer val
    """Validates shift bitwise by checking bit-shift operation behavior."""
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

    # keep shift
    # Regression for released branches: keep-left on signed values should preserve bits.
    x = Fxp(1, False, 8, 0, shifting='keep')
    prev = x.get_val()
    for i in range(x.n_word):
        assert (x << i)() == prev
        prev = 2 * prev
    assert (x << 8)() == 0

    # right unsigned
    x.set_val(128)
    prev = 128
    for i in range(x.n_word):
        assert (x >> i)() == prev
        prev = prev // 2
    assert (x >> 8)() == 0

    # left signed
    x = Fxp(1, True, 8, 0, shifting='keep')
    prev = 1
    for i in range(x.n_word - 1):
        assert (x << i)() == prev, f'{i}, {prev}'
        prev = 2 * prev
    assert (x << (x.n_word - 1))() == -128
    assert (x << x.n_word)() == 0

    x.set_val(113)
    for i in range(x.n_word):
        assert (x << i).bin() == x.bin()[i:] + (i * '0')

    x.set_val(-1)
    for i in range(x.n_word):
        assert (x << i).bin() == (x.n_word - i) * '1' + (i * '0')

    # right signed
    x.set_val(-128)
    for i in range(x.n_word):
        assert (x >> i).bin() == (i + 1) * '1' + (x.n_word - (i + 1)) * '0'
    assert (x >> 8).bin() == x.n_word * '1'

    # right signed sign extension from -1
    x.set_val(-1)
    for i in range(x.n_word + 2):
        assert (x >> i).bin() == x.n_word * '1'

    # None type
    x = Fxp(None, False, 8, 0, shifting='keep')
    for i in range(10):
        assert x << i == 0

    # arrays (1-D and N-D regression for PR #92 thread discussion)
    x = Fxp([1, 2, 4, 8], False, 8, 0, shifting='keep')
    assert np.all(x << 1 == x * 2)

    x.set_val([128])
    assert np.all(x << 1 == 0)

    x = Fxp([[1, 2], [4, 8]], False, 8, 0, shifting='keep')
    assert np.all(x << 1 == x * 2)

    # keep right-shift over arrays (including N-D and signed values)
    x = Fxp([[128, 64], [2, 1]], False, 8, 0, shifting='keep')
    assert np.array_equal((x >> 1)(), np.array([[64, 32], [1, 0]]))
    assert np.array_equal((x >> 8)(), np.zeros((2, 2), dtype=int))

    x = Fxp([[-128, -1], [-2, 3]], True, 8, 0, shifting='keep')
    assert np.array_equal((x >> 1)(), np.array([[-64, -1], [-1, 1]]))
    assert np.array_equal((x >> 8)(), np.array([[-1, -1], [-1, 0]]))

def test_invert():
    """Validates invert by checking binary representation/interpretation paths."""
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
    """Validates and by checking binary representation/interpretation paths."""
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
    """Validates or by checking binary representation/interpretation paths."""
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
    """Validates xor by checking binary representation/interpretation paths."""
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
    """Validates arrays by checking binary representation/interpretation paths, including multidimensional bitwise ops."""
    x = Fxp(None, True, 8, 4)
    y = Fxp(None, True, 8, 4)

    x(['0b00110101', '0b10101100'])
    y('0b11110000')

    z = x & y
    assert z.bin()[0] == '00110000'
    assert z.bin()[1] == '10100000'

    # 2D array with scalar masks (int and Fxp scalar) for multidimensional bitwise checks.
    x2 = Fxp(np.array([[0b00110101, 0b10101100], [0b11110000, 0b00001111]]), signed=False, n_word=8, n_frac=0)
    m2 = Fxp(0b11110000, signed=False, n_word=8, n_frac=0)

    exp_and_2d = np.array([[0b00110000, 0b10100000], [0b11110000, 0b00000000]])
    exp_or_2d = np.array([[0b11110101, 0b11111100], [0b11110000, 0b11111111]])
    exp_xor_2d = np.array([[0b11000101, 0b01011100], [0b00000000, 0b11111111]])
    exp_inv_2d = np.array([[0b11001010, 0b01010011], [0b00001111, 0b11110000]])

    assert np.array_equal((x2 & m2)(), exp_and_2d)
    assert np.array_equal((x2 | m2)(), exp_or_2d)
    assert np.array_equal((x2 ^ m2)(), exp_xor_2d)
    assert np.array_equal((~x2)(), exp_inv_2d)

    # 2D array-vs-array bitwise checks (same shape).
    m2_arr_data = np.array([[0b11110000, 0b11110000], [0b00001111, 0b00001111]])
    m2_arr = Fxp(m2_arr_data, signed=False, n_word=8, n_frac=0)

    assert np.array_equal((x2 & m2_arr)(), np.bitwise_and(np.array([[0b00110101, 0b10101100], [0b11110000, 0b00001111]]), m2_arr_data))
    assert np.array_equal((x2 | m2_arr)(), np.bitwise_or(np.array([[0b00110101, 0b10101100], [0b11110000, 0b00001111]]), m2_arr_data))
    assert np.array_equal((x2 ^ m2_arr)(), np.bitwise_xor(np.array([[0b00110101, 0b10101100], [0b11110000, 0b00001111]]), m2_arr_data))

    # 3D array with scalar masks for multidimensional broadcasting path.
    x3_data = np.arange(8).reshape(2, 2, 2)
    x3 = Fxp(x3_data, signed=False, n_word=4, n_frac=0)

    assert np.array_equal((x3 & 0b0011)(), x3_data & 0b0011)
    assert np.array_equal((x3 | 0b0101)(), x3_data | 0b0101)
    assert np.array_equal((x3 ^ 0b0110)(), x3_data ^ 0b0110)

    m3_data = np.array([[[0b0011], [0b0101]]])
    m3 = Fxp(m3_data, signed=False, n_word=4, n_frac=0)

    assert np.array_equal((x3 & m3)(), np.bitwise_and(x3_data, m3_data))
    assert np.array_equal((x3 | m3)(), np.bitwise_or(x3_data, m3_data))
    assert np.array_equal((x3 ^ m3)(), np.bitwise_xor(x3_data, m3_data))
    assert np.array_equal((~x3)(), ((~x3_data) & 0b1111))

def test_operations_with_combinations():
    """Exhaustively compare Fxp-vs-Fxp arithmetic against Python numeric results over mixed signed values."""
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
    """Exhaustively compare mixed Fxp/constant arithmetic against Python numeric results across value grids."""
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

def _overflow_stress_boundary_frac_operands():
    # Stress scalar scaling close to native integer boundary (n_frac = _n_word_max - 1).
    """Validates  overflow stress boundary frac operands by checking overflow/wrap/saturate behavior, NumPy interoperability, scale/bias conversion behavior."""
    x = Fxp(np.array([1.0, -2.0]), signed=True, n_word=16, n_frac=0)
    y_n_word = int(fxp._n_word_max)
    y_n_frac = y_n_word - 1
    y = Fxp(np.array([0.5, 0.5]), signed=True, n_word=y_n_word, n_frac=y_n_frac)
    return x, y

def _overflow_stress_truediv_operands():
    # Keep x simple and force a very large scale shift through y format.
    """Validates  overflow stress truediv operands by checking overflow/wrap/saturate behavior, NumPy interoperability, scale/bias conversion behavior."""
    x = Fxp(np.array([1.0, -2.0]), signed=True, n_word=16, n_frac=0)
    y_n_word = int(fxp._n_word_max + 8)
    y_n_int = 4
    y_n_frac = y_n_word - 1 - y_n_int
    y = Fxp(np.array([1.0, 0.5]), signed=True, n_word=y_n_word, n_frac=y_n_frac)
    return x, y

def test_add_raw_intermediate_overflow():
    """Validates add raw intermediate overflow by checking overflow/wrap/saturate behavior, NumPy interoperability."""
    x, y = _overflow_stress_boundary_frac_operands()
    z = x + y
    assert np.all(z() == np.array([1.5, -1.5]))

def test_sub_raw_intermediate_overflow():
    """Validates sub raw intermediate overflow by checking overflow/wrap/saturate behavior, NumPy interoperability."""
    x, y = _overflow_stress_boundary_frac_operands()
    z = x - y
    assert np.all(z() == np.array([0.5, -2.5]))

def test_mod_raw_intermediate_overflow():
    """Validates mod raw intermediate overflow by checking overflow/wrap/saturate behavior, NumPy interoperability."""
    x, y = _overflow_stress_boundary_frac_operands()
    z = x % y
    assert np.all(z() == np.array([0.0, 0.0]))

def test_truediv_raw_intermediate_overflow():
    # Keep output precision small but force a huge intermediate shift in raw division.
    """Validates truediv raw intermediate overflow by checking overflow/wrap/saturate behavior, NumPy interoperability, bit-shift operation behavior."""
    x, y = _overflow_stress_truediv_operands()
    z = x / y
    assert np.all(z() == np.array([1.0, -4.0]))

def test_floordiv_raw_intermediate_overflow_with_high_frac_out():
    """Validates floordiv raw intermediate overflow with high frac out by checking overflow/wrap/saturate behavior, NumPy interoperability."""
    x, y = _overflow_stress_truediv_operands()
    out = Fxp(np.zeros(2), signed=True, n_word=96, n_frac=63)
    z = fxp.floordiv(x, y, out=out)
    assert np.all(z() == np.array([1.0, -4.0]))

def test_pow():
    """Validate power behavior for integer, fractional, signed/unsigned, scalar, and vectorized exponent cases."""
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
    """Validates scaled by checking scale/bias conversion behavior."""
    x = Fxp(10.5, True, 16, 8, scale=2, bias=1)

    assert x() == 10.5
    
    assert x + 2 == 12.5
    assert x - 2.5 == 8.0
    assert x * 3 == 31.5
    assert x / 2 == 5.25

def test_abs():
    """Verify absolute-value operator returns non-negative represented values for signed inputs."""
    x = Fxp(-3.5, True, 32, 16)

    assert x() == -3.5
    assert abs(x)() == 3.5

    x = Fxp(3.5, True, 32, 16)
    assert abs(x)() == 3.5
    

def test_bitwise_large_word_scalar_and_arrays():
    """Validates >64-bit bitwise ops for scalar and array cases across all bitwise operators."""
    n_word = 80
    full_mask = (1 << n_word) - 1

    x_val = (1 << 79) + (1 << 40) + 0x12345
    y_val = (1 << 78) + (1 << 40) + 0x00FF00FF00FF
    m_val = (1 << 79) + (1 << 12) + 0xAAAA

    x = Fxp(x_val, signed=False, n_word=n_word, n_frac=0)
    y = Fxp(y_val, signed=False, n_word=n_word, n_frac=0)

    # Scalar Fxp-vs-Fxp
    assert (x & y)() == (x_val & y_val)
    assert (x | y)() == (x_val | y_val)
    assert (x ^ y)() == (x_val ^ y_val)
    assert (~x)() == (full_mask - x_val)

    # Scalar Fxp-vs-int and reversed int-vs-Fxp paths.
    assert (x & m_val)() == (x_val & m_val)
    assert (x | m_val)() == (x_val | m_val)
    assert (x ^ m_val)() == (x_val ^ m_val)
    assert (m_val & x)() == (m_val & x_val)
    assert (m_val | x)() == (m_val | x_val)
    assert (m_val ^ x)() == (m_val ^ x_val)

    # Array (same-shape) Fxp-vs-Fxp
    xa_data = np.array(
        [[x_val, y_val], [full_mask, 0]],
        dtype=object,
    )
    ya_data = np.array(
        [[m_val, x_val], [y_val, full_mask]],
        dtype=object,
    )

    xa = Fxp(xa_data, signed=False, n_word=n_word, n_frac=0)
    ya = Fxp(ya_data, signed=False, n_word=n_word, n_frac=0)

    and_expected = np.frompyfunc(lambda a, b: a & b, 2, 1)(xa_data, ya_data)
    or_expected = np.frompyfunc(lambda a, b: a | b, 2, 1)(xa_data, ya_data)
    xor_expected = np.frompyfunc(lambda a, b: a ^ b, 2, 1)(xa_data, ya_data)
    inv_expected = np.frompyfunc(lambda a: full_mask - a, 1, 1)(xa_data)

    assert np.array_equal((xa & ya)(), and_expected)
    assert np.array_equal((xa | ya)(), or_expected)
    assert np.array_equal((xa ^ ya)(), xor_expected)
    assert np.array_equal((~xa)(), inv_expected)

    # Array broadcast path (2x2 with 1x2)
    yb_data = np.array([[m_val, y_val]], dtype=object)
    yb = Fxp(yb_data, signed=False, n_word=n_word, n_frac=0)
    yb_broadcast = np.broadcast_to(yb_data, xa_data.shape)

    assert np.array_equal((xa & yb)(), np.frompyfunc(lambda a, b: a & b, 2, 1)(xa_data, yb_broadcast))
    assert np.array_equal((xa | yb)(), np.frompyfunc(lambda a, b: a | b, 2, 1)(xa_data, yb_broadcast))
    assert np.array_equal((xa ^ yb)(), np.frompyfunc(lambda a, b: a ^ b, 2, 1)(xa_data, yb_broadcast))
