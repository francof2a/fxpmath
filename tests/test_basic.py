import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import fxpmath as fxp
from fxpmath.objects import Fxp

import numpy as np

def test_temp():
    pass

def test_instances():
    x = Fxp()
    assert type(x) == Fxp
    assert x.dtype == 'fxp-s16/15'
    assert x() == 0

    x = Fxp(-0.5)
    assert x() == -0.5
    assert x.signed == True
    assert x.n_frac == 1
    assert x.n_int == 0
    assert x.n_word == 2

    x = Fxp(-3)
    assert x() == -3
    assert x.signed == True
    assert x.n_frac == 0
    assert x.n_int == 2
    assert x.n_word == 3

    x = Fxp(-3.5, True, 8, 4)
    assert x() == -3.5
    assert x.signed == True
    assert x.n_frac == 4
    assert x.n_int == 3
    assert x.n_word == 8

    x = Fxp(7.5, False, 8, 4)
    assert x() == 7.5
    assert x.signed == False
    assert x.n_frac == 4
    assert x.n_int == 4
    assert x.n_word == 8

    x = Fxp(7.5, dtype='UQ4.4')
    assert x() == 7.5
    assert x.signed == False
    assert x.n_frac == 4
    assert x.n_int == 4
    assert x.n_word == 8
    
    x = Fxp(7.5, dtype='U4.4')
    assert x() == 7.5
    assert x.signed == False
    assert x.n_frac == 4
    assert x.n_int == 4
    assert x.n_word == 8
    
    x = Fxp(7.5, dtype='fxp-u8/4')
    assert x() == 7.5
    assert x.signed == False
    assert x.n_frac == 4
    assert x.n_int == 4
    assert x.n_word == 8
    
    x = Fxp(7.5, False, 8, 4)
    assert x() == 7.5
    assert x.signed == False
    assert x.n_frac == 4
    assert x.n_int == 4
    assert x.n_word == 8
    
    x = Fxp(7.5, True, n_frac=4, n_int=6)
    assert x() == 7.5
    assert x.signed == True
    assert x.n_frac == 4
    assert x.n_int == 6
    assert x.n_word == 11

    x = Fxp(7.5, dtype='Q7.4')
    assert x() == 7.5
    assert x.signed == True
    assert x.n_frac == 4
    assert x.n_int == 6
    assert x.n_word == 11

    x = Fxp(7.5, dtype='S7.4')
    assert x() == 7.5
    assert x.signed == True
    assert x.n_frac == 4
    assert x.n_int == 6
    assert x.n_word == 11

    x = Fxp(7.5, dtype='fxp-s11/4')
    assert x() == 7.5
    assert x.signed == True
    assert x.n_frac == 4
    assert x.n_int == 6
    assert x.n_word == 11
    
    x = Fxp(3, False)
    assert x() == 3
    assert x.signed == False
    assert x.n_frac == 0
    assert x.n_int == 2
    assert x.n_word == 2

    x = Fxp(1.75)
    assert x() == 1.75
    assert x.signed == True
    assert x.n_frac == 2
    assert x.n_int == 1
    assert x.n_word == 4

    x = Fxp(1.75, False)
    assert x() == 1.75
    assert x.signed == False
    assert x.n_frac == 2
    assert x.n_int == 1
    assert x.n_word == 3

    x = Fxp([-1, 0, 1, 2, 3], True, n_word=16, n_frac=4)
    assert (x() == np.array([-1, 0, 1, 2, 3])).all()

    x = Fxp(0.25 + 1j*15)
    assert x() == 0.25 + 1j*15
    assert x.dtype == 'fxp-s7/2-complex'
    assert x.signed == True
    assert x.n_frac == 2
    assert x.n_int == 4
    assert x.n_word == 7

    x = Fxp('0b00000.01+0b01111.00j')
    assert x() == 0.25 + 1j*15

    x = Fxp('0b00000.01+0b11111.11j')
    assert x() == 0.25 - 1j*0.25

    x = Fxp('0b00000.01-0b0000.10j')
    assert x() == 0.25 - 1j*0.5

    x = Fxp([[1.5, 2.25], [-0.125, -3.75]])
    assert (x() == np.array([[1.5, 2.25], [-0.125, -3.75]])).all()

    x = Fxp([['0b1100', '0b0110'], ['0b0000', '0b1111']], signed=True, n_frac=2)
    assert (x() == np.array([[-1.0, 1.5], [0.0, -0.25]])).all()

    # Fxp from a Fxp
    x = Fxp(-1.75, dtype='fxp-s8/4')
    y = Fxp(x)
    assert x() == y()

    y = Fxp(x, like=x)
    assert x() == y() and x.dtype == y.dtype

    y = Fxp(x, signed=False)
    assert x() != y() and y() == 0

    x = Fxp(1.75, dtype='fxp-u8/4')
    y = Fxp(x)
    assert x() == y()

    x1 = Fxp(4, False, 9, 3)
    x2 = Fxp(5, False, 9, 3)
    cast = Fxp(None, True, 9, 3)
    y = cast(x1) - cast(x2)

def test_signed():
    # signed
    x_fxp = Fxp(0.0, True, 8, 7)

    x_fxp(0.5)
    assert x_fxp() == 0.5

    x_fxp(-0.5)
    assert x_fxp() == -0.5 

    # unsigned
    x_fxp = Fxp(0.0, False, 8, 7)

    x_fxp(0.5)
    assert x_fxp() == 0.5

    x_fxp(-0.5)
    assert x_fxp() == 0.0

def test_misc_values():
    # huges
    x = Fxp(2**31 - 1)
    assert x() == 2**31 - 1
    x = Fxp(-2**31)
    assert x() == -2**31

    x = Fxp(2**63 - 1)
    assert x() == 2**63 - 1
    x = Fxp(-2**63)
    assert x() == -2**63

    x = Fxp(2**32 - 1, signed=False)
    assert x() == 2**32 - 1
    x = Fxp(-2**32, signed=False)
    assert x() == 0

    x = Fxp(2**64 - 1, signed=False)
    assert x() == 2**64 - 1
    x = Fxp(-2**63, signed=False)
    assert x() == 0

    x = Fxp(2**64, n_word_max=128)
    assert x() == 2**64

    x = Fxp(2.0**64  - 1, signed=False, n_word=128, n_frac=64)
    assert x() == 2.0**64 - 1

    x = Fxp(2**128  - 1, signed=False, n_word=128, n_frac=0)
    assert x() == 2**128 - 1

    x = Fxp(2.0**32  - 1, signed=False, n_word=128, n_frac=96)
    assert x() == 2.0**32 - 1

    x = Fxp([2**-64, -2**96, 2**32], True, 256, 128)
    assert np.isclose(x()[0], 2**-64, rtol=2**-64)
    assert x()[1] == -2**96
    assert x()[2] == 2**32

def test_base_representations():
    x = Fxp(0.0, True, 8, 4)

    # decimal positive
    x(2.5)
    assert x.bin() == '00101000'
    assert x.bin(frac_dot=True) == '0010.1000'
    assert x.bin(prefix='0b') == '0b00101000'
    assert x.bin(prefix=True) == '0b00101000'
    assert x.hex() == '0x28'
    assert x.hex(padding=False) == '0x28'
    assert x.base_repr(2) == '101000'
    assert x.base_repr(16) == '28'
    
    # decimal negative
    x(-7.25)
    assert x.bin() == '10001100'
    assert x.bin(frac_dot=True) == '1000.1100'
    assert x.bin(frac_dot=True, prefix='b') == 'b1000.1100'
    assert x.hex() == '0x8C'
    assert x.hex(padding=False) == '0x8C'
    assert x.base_repr(2) == '-1110100'
    assert x.base_repr(2, frac_dot=True) == '-111.0100'
    assert x.base_repr(16) == '-74'

    # complex
    x(1.5 + 1j*0.75)
    assert x.bin() == '00011000+00001100j'
    assert x.bin(frac_dot=True) == '0001.1000+0000.1100j'
    assert x.bin(frac_dot=True, prefix=True) == '0b0001.1000+0b0000.1100j'
    assert x.hex() == '0x18+0x0Cj'
    assert x.hex(padding=False) == '0x18+0xCj'
    assert x.base_repr(2) == '11000+1100j'
    assert x.base_repr(16) == '18+Cj'  

    x(1.5 - 1j*0.75)
    assert x.bin() == '00011000+11110100j'
    assert x.hex() == '0x18+0xF4j'
    assert x.hex(padding=False) == '0x18+0xF4j'
    assert x.base_repr(2) == '11000-1100j'
    assert x.base_repr(16) == '18-Cj'

    x(-1.5 + 1j*0.75)
    assert x.bin() == '11101000+00001100j'
    assert x.hex() == '0xE8+0x0Cj'
    assert x.hex(padding=False) == '0xE8+0xCj'
    assert x.base_repr(2) == '-11000+1100j'
    assert x.base_repr(16) == '-18+Cj' 

    # padding
    x = Fxp(3.125, True, 31, 5)
    y = Fxp(-3.125, True, 31, 5)

    assert x.hex() == '0x00000064'
    assert y.hex() == '0x7FFFFF9C'

    x = Fxp(3.125, True, 32, 5)
    y = Fxp(-3.125, True, 32, 5)

    assert x.hex() == '0x00000064'
    assert y.hex() == '0xFFFFFF9C'

    x = Fxp(3.125, True, 33, 5)
    y = Fxp(-3.125, True, 33, 5)

    assert x.hex() == '0x000000064'
    assert y.hex() == '0x1FFFFFF9C'

    # arrays
    arr_fxp = Fxp(np.array([[1, 2]]))
    assert np.all(arr_fxp.bin() == np.array(['001', '010']))
    assert np.all(arr_fxp.hex() == np.array(['0x1', '0x2']))
    assert np.all(arr_fxp.base_repr(base=2) == np.array(['1', '10']))
    assert np.all(arr_fxp.base_repr(base=10) == np.array(['1', '2']))
    assert np.all(arr_fxp.base_repr(base=10) == np.array(['1', '2']))

    arr_fxp = Fxp(np.array([[15, -16], [-1, 0]]))
    assert np.all(arr_fxp.bin() == np.array([['01111', '10000'], ['11111', '00000']]))
    assert np.all(arr_fxp.hex() == np.array([['0x0F', '0x10'], ['0x1F', '0x00']]))
    assert np.all(arr_fxp.base_repr(base=2) == np.array([['1111', '-10000'], ['-1', '0']]))
    assert np.all(arr_fxp.base_repr(base=10) == np.array([['15', '-16'], ['-1', '0']]))
    assert np.all(arr_fxp.base_repr(base=16) == np.array([['F', '-10'], ['-1', '0']]))


def test_like():
    ref = Fxp(0.0, True, 16, 4)
    x = Fxp(4.5, like=ref)
    assert x is not ref
    assert x() == 4.5
    assert x.n_word == ref.n_word
    assert x.n_frac == ref.n_frac
    assert x.signed == ref.signed

def test_kwargs():
    x = Fxp(-2.125, True, 16, 4, overflow='wrap')
    assert x.config.overflow == 'wrap'
    y = Fxp(3.2, True, 16, 8, rounding='fix')
    assert y.config.rounding == 'fix'

def test_strvals():
    x = Fxp('0b0110')
    assert x() == 6
    x = Fxp('0b110', True, 4, 0)
    assert x() == -2
    x = Fxp('0b110', False, 4, 0)
    assert x() == 6
    x = Fxp('0b0110.01', True, 8)
    assert x() == 6.25

    x = Fxp(0.0, True, 8, 4)
    x('0x8c')
    assert x() == -7.25

def test_saturate():
    x = Fxp(0.0, True, 8, 2)
    assert x.upper == 31.75
    assert x.lower == -32.00
    assert x.status['overflow'] == False
    assert x.status['underflow'] == False

    assert x(32.00) == 31.75
    assert x.status['overflow'] == True
    assert x.status['underflow'] == False

    x.reset()
    assert x.status['overflow'] == False
    assert x.status['underflow'] == False

    assert x(-32.25) == -32.00
    assert x.status['overflow'] == False
    assert x.status['underflow'] == True

    assert x(32.00) == 31.75
    assert x.status['overflow'] == True
    assert x.status['underflow'] == True    

def test_rounding():
    # trunc
    x = Fxp(None, True, 8, 2, rounding='trunc')
    vi = [0.00, 1.00, 1.24, 1.25, 1.26, 1.49, 1.50]
    vo = [0.00, 1.00, 1.00, 1.25, 1.25, 1.25, 1.50]
    for i, o in zip(vi, vo):
        assert x(i) == o
        assert x(-i) == -o

    # ceil
    x = Fxp(None, True, 8, 2, rounding='ceil')
    vi = [0.00, 1.00, 1.24, 1.25, 1.26, 1.49, 1.50, -1.00, -1.24, -1.25, -1.26, -1.49, -1.50]
    vo = [0.00, 1.00, 1.25, 1.25, 1.50, 1.50, 1.50, -1.00, -1.00, -1.25, -1.25, -1.25, -1.50]
    for i, o in zip(vi, vo):
        assert x(i) == o

    # floor
    x = Fxp(None, True, 8, 2, rounding='floor')
    vi = [0.00, 1.00, 1.24, 1.25, 1.26, 1.49, 1.50, -1.00, -1.24, -1.25, -1.26, -1.49, -1.50]
    vo = [0.00, 1.00, 1.00, 1.25, 1.25, 1.25, 1.50, -1.00, -1.25, -1.25, -1.50, -1.50, -1.50]
    for i, o in zip(vi, vo):
        assert x(i) == o
    
    # fix
    x = Fxp(None, True, 8, 2, rounding='fix')
    vi = [0.00, 1.00, 1.24, 1.25, 1.26, 1.49, 1.50, -1.00, -1.24, -1.25, -1.26, -1.49, -1.50]
    vo = [0.00, 1.00, 1.00, 1.25, 1.25, 1.25, 1.50, -1.00, -1.00, -1.25, -1.25, -1.25, -1.50]
    for i, o in zip(vi, vo):
        assert x(i) == o

    # around
    x = Fxp(None, True, 8, 2, rounding='around')
    vi = [0.00, 1.00, 1.24, 1.25, 1.26, 1.49, 1.50, -1.00, -1.24, -1.25, -1.26, -1.49, -1.50]
    vo = [0.00, 1.00, 1.25, 1.25, 1.25, 1.50, 1.50, -1.00, -1.25, -1.25, -1.25, -1.50, -1.50]
    for i, o in zip(vi, vo):
        assert x(i) == o

def test_scaling():
    x = Fxp(4.5, scale=2.0, bias=-1.5)
    assert x() == 4.5
    assert x.n_word == 3
    assert x.n_frac == 0
    assert x.upper == 4.5
    assert x.lower == -9.5
    assert x.precision == 2.0

    x = Fxp(1003, False, scale=0.5, bias=1000)
    assert x() == 1003
    assert x.n_word == 3
    assert x.n_frac == 0
    assert x.upper == 1003.5
    assert x.lower == 1000
    assert x.precision == 0.5   

    x = Fxp(10128.5, signed=False, n_word=12, scale=1, bias=10000)
    assert x() == 10128.5
    assert x.n_word == 12
    assert x.n_frac == 1
    assert x.upper == 12047.5
    assert x.lower == 10000.0
    assert x.precision == 0.5
    assert x.get_status(str) == ''   

def test_wrap():
    x = Fxp(3.75, False, 4, 2, overflow='wrap')
    assert x() == 3.75
    assert x.status['overflow'] == False
    assert x.status['underflow'] == False

    x(4.0)
    assert x() == 0.0
    assert x.status['overflow'] == True
    assert x.status['underflow'] == False

    x.reset()
    x(-0.25)
    assert x() == 3.75
    assert x.status['overflow'] == False
    assert x.status['underflow'] == True

    x = Fxp(3.75, True, 5, 2, overflow='wrap')
    assert x() == 3.75
    assert x.status['overflow'] == False
    assert x.status['underflow'] == False

    x(4.0)
    assert x() == -4.0
    assert x.status['overflow'] == True
    assert x.status['underflow'] == False

    x.reset()
    x(-4.25)
    assert x() == 3.75
    assert x.status['overflow'] == False
    assert x.status['underflow'] == True

def test_init_by_raw():
    x = Fxp(16, True, 8, 4, raw=True)
    assert x() == 1.0

    x = Fxp(16, None, None, 4, raw=True)
    assert x() == 1.0

    x = Fxp('0b1111', n_frac=1, raw=True) 
    assert x() == -0.5

def test_resize():
    x = Fxp(12.5, True, 16, 4)
    assert x() == 12.5

    # same sizes
    x.resize(n_word=16)
    assert x() == 12.5
    assert x.signed == True
    assert x.n_word == 16
    assert x.n_frac == 4

    x.resize(n_frac=4)
    assert x() == 12.5
    assert x.signed == True
    assert x.n_word == 16
    assert x.n_frac == 4

    # smaller sizes
    x.resize(n_word=12)
    assert x() == 12.5
    assert x.signed == True
    assert x.n_word == 12
    assert x.n_frac == 4   

    x.resize(n_frac=2)
    assert x() == 12.5
    assert x.signed == True
    assert x.n_word == 12
    assert x.n_frac == 2

    # return to original size
    x.resize(dtype='S12.4')
    assert x() == 12.5
    assert x.signed == True
    assert x.n_word == 16
    assert x.n_frac == 4

    # bigger sizes
    x.resize(n_word=24, n_frac=8)
    assert x() == 12.5
    assert x.signed == True
    assert x.n_word == 24
    assert x.n_frac == 8

    # to unsigned
    x.resize(dtype='U12.4')
    assert x() == 12.5
    assert x.signed == False
    assert x.n_word == 16
    assert x.n_frac == 4

    # force truncation
    x.resize(dtype='S12.4') # return to original size

    x.resize(dtype='S4.2')
    assert x() == x.upper
    assert x() == 7.75

    # force wrapping by overflow
    x = Fxp(12.5, True, 16, 4)

    x.overflow = 'wrap'
    x.resize(dtype='S4.1')
    assert x() == 12.5 - x.upper - x.precision + x.lower 
    assert x() == -3.5
        