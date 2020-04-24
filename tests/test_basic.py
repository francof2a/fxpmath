import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import fxpmath as fxp
from fxpmath.objects import Fxp


def test_temp():
    x = Fxp(0.5, True, 8, 7)
    assert x.astype(float) == 0.5

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

def test_base_representations():
    x = Fxp(0.0, True, 8, 4)

    # decimal positive
    x(2.5)
    assert x.bin() == '00101000'
    assert x.hex() == '0x28'
    assert x.base_repr(2) == '101000'
    assert x.base_repr(16) == '28'
    
    # decimal negative
    x(-7.25)
    assert x.bin() == '10001100'
    assert x.hex() == '0x8c'
    assert x.base_repr(2) == '-1110100'
    assert x.base_repr(16) == '-74'

    # complex
    x(1.5 + 1j*0.75)
    assert x.bin() == '00011000+00001100j'
    assert x.hex() == '0x18+0xcj'
    assert x.base_repr(2) == '11000+1100j'
    assert x.base_repr(16) == '18+Cj'  

    x(1.5 - 1j*0.75)
    assert x.bin() == '00011000+11110100j'
    assert x.hex() == '0x18+0xf4j'
    assert x.base_repr(2) == '11000-1100j'
    assert x.base_repr(16) == '18-Cj'

    x(-1.5 + 1j*0.75)
    assert x.bin() == '11101000+00001100j'
    assert x.hex() == '0xe8+0xcj'
    assert x.base_repr(2) == '-11000+1100j'
    assert x.base_repr(16) == '-18+Cj' 