import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import fxpmath as fxp
from fxpmath.objects import Fxp

import numpy as np

def test_issue_9_v0_3_6():
    M = 24
    N = 16
    A_fxp = Fxp(np.zeros((M,N)), True, 16, 8)

    # indexed element returned as Fxp object
    assert isinstance(A_fxp[0], Fxp)

    # accept an Fxp object like input value
    x = Fxp(4, True, 8, 2)
    A_fxp[0,0] = x
    assert A_fxp[0,0]() == 4

    B_fxp = Fxp(x, like=A_fxp)
    assert B_fxp() == 4

    C_fxp = Fxp(x)
    assert C_fxp() == 4

def test_issue_10_v0_3_6():
    x = Fxp(1.5, True, 256, 64)
    assert x() == 1.5

def test_issue_11_v0_3_6():
    try:
        val = np.float128(1.5)
    except:
        val = None
    
    if val is not None:
        x = Fxp(val, True, 256, 64)
        assert x() == 1.5

    # x = Fxp(np.float128(1.5), True, 256, 64)
    # assert x() == 1.5

def test_issue_14_v0_3_7():
    # d = Fxp('0b000.00000000000000000001011101010110101011101010101010101010101010101010101010101001010101010101010101001010101001010101010101010', 
    #         True, n_word=128, n_frac=125, rounding='around')
    # assert d.bin(frac_dot=True) == '000.00000000000000000001011101010110101011101010101010101010101010101010101010101001010101010101010101001010101001010101010101010'

    d = Fxp('0b00000000000000000000001011101010110101011101010101010101010101010101010101010101001010101010101010101001010101001010101010101010', 
            True, n_word=128, n_frac=125, rounding='around', raw=True)
    assert d.bin() == '00000000000000000000001011101010110101011101010101010101010101010101010101010101001010101010101010101001010101001010101010101010'
 

def test_issue_15_v0_3_7():
    x = Fxp('0xb', True, 10, 4)
    assert x.hex() == '0x00B'


def test_issue_17_v0_3_7():
    a = Fxp(15, signed=False)
    b = a ** Fxp(2)

    assert b() == 15**2

def test_issue_19_v0_3_7():
    DW=12
    DATA_FXPTYPE = Fxp(None, signed=True, n_word=DW, n_frac=DW-1)

    a = Fxp(np.zeros(2, dtype=complex), like=DATA_FXPTYPE)
    c = Fxp(complex(0,0), like=DATA_FXPTYPE)
    b = Fxp(0.5-0.125j, like=DATA_FXPTYPE)

    # print(c.get_val() + b.get_val())
    c.equal(c+b)
    
    # c.info()
    assert c() == 0.5-0.125j

    # print(a[0].get_val() + b.get_val())
    # a[0].equal(a[0]+b) # not supported
    a[0] = a[0]+b
    
    # a[0].info()
    assert a[0]() == 0.5-0.125j

def test_issue_20_v0_3_8():
    x = Fxp(0, signed=True, n_word = 4, n_frac = 0, overflow='wrap')

    assert x(-30) == 2
    assert x(-8) == -8
    assert x(-9) == 7
    assert x(7) == 7
    assert x(8) == -8

def test_issue_21_v0_3_8():
    a = [1, 2, 3]
    b = [0, 1, 0]
    assert (np.inner(a, b) == 2)

    na = np.array([1, 2, 3])
    nb = np.array([0, 1, 0])
    assert (np.inner(na, nb) == np.inner(a, b)).all()

    fa = Fxp([1, 2, 3])
    fb = Fxp([0, 1, 0])
    z = np.inner(fa, fb)
    assert (np.inner(fa, fb)() == np.inner(a, b)).all()