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