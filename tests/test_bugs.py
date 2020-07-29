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

def test_bugs_0_3_0():
    # fail in Win32 because numpy astype(int) behavior
    x = Fxp(4.001)
    assert x() == 4.001

def test_bugs_0_3_2():
    # fail in Win32 because numpy astype(int) behavior
    x = Fxp(1.25, False, 3, 1)
    assert (x >> 1)() == 0.5

    # wrap error
    x = Fxp(4.5, False, 3, 1, overflow='wrap')
    assert x() == 0.5
    assert x(4.0) == 0.0
    assert x(5.0) == 1.0
    assert (x([3.5, 4.0, 4.5, 5.0])() == np.array([3.5, 0.0, 0.5, 1.0])).all()

def test_bugs_0_3_3():
    # wrap error
    x = Fxp(12.5, False, 11, 8, overflow='wrap')
    assert x() == 4.5

def test_bugs_0_3_4():
    # wrap error for ndarrays
    x = Fxp([[1, 1]], False, 17 + 3, 9, overflow='wrap')
    assert (x() == np.array([[1, 1]])).all()