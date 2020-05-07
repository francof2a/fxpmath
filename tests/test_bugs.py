import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import fxpmath as fxp
from fxpmath.objects import Fxp

import numpy as np


def test_bugs_0_2_1():
    a = Fxp(0.5)
    b = Fxp(0.5)
    c = a * b
    assert c() == 0.25
    assert c.n_frac == 2
    assert c.n_word == 4

    a = Fxp(1.75, n_int=1, n_frac=2)
    b = Fxp(1.75, n_int=1, n_frac=2)
    c = a * b
    assert c() == 3.0625
    assert c.n_frac == 4
    assert c.n_word == 8

    a = Fxp(-2.00, n_int=1, n_frac=2)
    b = Fxp(-2.00, n_int=1, n_frac=2)
    c = a * b
    assert c() == 4.00
    assert c.n_frac == 4
    assert c.n_word == 8

def test_bugs_0_2_2():
    x = Fxp('0b1100')
    assert x() == -4
    assert x.n_word == 4
    assert x.signed == True
    assert x.n_frac == 0

    x = Fxp('0b11.00')
    assert x() == -1.0
    assert x.n_word == 4
    assert x.signed == True
    assert x.n_frac == 2