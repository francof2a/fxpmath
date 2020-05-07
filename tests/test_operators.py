import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import fxpmath as fxp
from fxpmath.objects import Fxp

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
    assert (x >> 6)() == 0

    # float val
    x = Fxp(24.25, True, 8, 2)
    #left
    assert (x << 1)() == 48.5
    assert (x << 4)() == 388.0
    #right
    x = Fxp(24.5, True, 8, 2)
    assert (x >> 1)() == 12.25
    assert (x >> 2)() == 6.0

    # negative
    x = Fxp(-24.25, True, 8, 2)
    #left
    assert (x << 1)() == -48.5
    assert (x << 4)() == -388.0
    #right
    x = Fxp(-24.5, True, 8, 2)
    assert (x >> 1)() == -12.25
    assert (x >> 2)() == -6.25

    # trunc left shift
    x = Fxp(32, True, 8, 0, shifting='trunc')
    assert (x << 1)() == 64
    assert (x << 2)() == x.upper

