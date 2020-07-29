import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import fxpmath as fxp
from fxpmath.utils import *
from fxpmath.functions import *

import numpy as np

def test_sum():
    vals = np.array([-2, -1, 0, 1, 2, 3, 4])

    x = Fxp(vals, True, 32, 2)
    y = fxp.sum(x)
    assert (y() == np.sum(vals)).all()

    y = fxp.sum(x, sizes='same_sizes')
    assert (y() == np.sum(vals)).all()
    assert y.n_word == x.n_word
    assert y.n_frac == x.n_frac

    y = fxp.sum(x, sizes='tight_sizes')
    assert (y() == np.sum(vals)).all()

    z = Fxp(None, True, 16, 4)
    y = fxp.sum(x, out=z)
    assert (y() == np.sum(vals)).all()
    assert (z() == np.sum(vals)).all()
    assert y.dtype == z.dtype

    y = fxp.sum(x, dtype=z.dtype)
    assert (y() == np.sum(vals)).all()
    assert y.dtype == z.dtype

    y = fxp.sum(x, dtype='fxp-s16/2')
    assert (y() == np.sum(vals)).all()
    assert y.signed == True
    assert y.n_word == 16
    assert y.n_frac == 2

    vals = np.array([
        [-2, -1, 0], 
        [1, 2, 3]])
    x = Fxp(vals, True, 32, 2)
    
    y = fxp.sum(x, axis=0)
    assert (y() == np.sum(vals, axis=0)).all()

    y = fxp.sum(x, axis=1)
    assert (y() == np.sum(vals, axis=1)).all()
