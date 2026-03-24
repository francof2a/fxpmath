import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import fxpmath as fxp
from fxpmath.utils import *
from fxpmath.functions import *

import numpy as np

def test_fxp_sum():
    """Validates fxp sum by checking NumPy reduction interoperability, NumPy interoperability, dtype parsing and conversion behavior."""
    vals = np.array([-2, -1, 0, 1, 2, 3, 4])

    x = Fxp(vals, True, 16, 2)
    y = fxp.fxp_sum(x)
    assert (y() == np.sum(vals)).all()

    y = fxp.fxp_sum(x, sizes='same_sizes')
    assert (y() == np.sum(vals)).all()
    assert y.n_word == x.n_word
    assert y.n_frac == x.n_frac

    y = fxp.fxp_sum(x, sizes='tight_sizes')
    assert (y() == np.sum(vals)).all()

    z = Fxp(None, True, 16, 4)
    y = fxp.fxp_sum(x, out=z)
    assert (y() == np.sum(vals)).all()
    assert (z() == np.sum(vals)).all()
    assert y.dtype == z.dtype

    y = fxp.fxp_sum(x, dtype=z.dtype)
    assert (y() == np.sum(vals)).all()
    assert y.dtype == z.dtype

    y = fxp.fxp_sum(x, dtype='fxp-s16/2')
    assert (y() == np.sum(vals)).all()
    assert y.signed == True
    assert y.n_word == 16
    assert y.n_frac == 2

    vals = np.array([
        [-2, -1, 0], 
        [1, 2, 3]])
    x = Fxp(vals, True, 16, 2)
    
    y = fxp.fxp_sum(x, axis=0)
    assert (y() == np.sum(vals, axis=0)).all()

    y = fxp.fxp_sum(x, axis=1)
    assert (y() == np.sum(vals, axis=1)).all()

def test_from_bin():
    """Validates from bin by checking binary string parsing and inferred sizing, binary representation/interpretation paths, dtype parsing and conversion behavior."""
    x = from_bin('0', signed=False)
    assert x() == 0

    x = fxp.from_bin('1', signed=False)
    assert x() == 1

    x = fxp.from_bin('011')
    assert x() == 3

    x = fxp.from_bin('111')
    assert x() == -1

    x = fxp.from_bin('111', signed=False)
    assert x() == 7

    x = fxp.from_bin('1.11')
    assert x() == -0.25

    x = fxp.from_bin('0b1.11', signed=False)
    assert x() == 1.75

    x = fxp.from_bin('0.11')
    assert x() == 0.75

    x = fxp.from_bin('01100100.01')
    assert x() == 100.25
    assert x.n_word == 10 and x.n_frac == 2

    x = fxp.from_bin('01100100.01', dtype='fxp-s16/4')
    assert x() == 100.25
    assert x.n_word == 16 and x.n_frac == 4

def test_boundary_precision_scaling_in_raw_functions():
    """Validates boundary precision scaling in raw functions by checking NumPy reduction interoperability, NumPy cumulative reduction interoperability, NumPy dot-product interoperability."""
    n_frac = fxp._n_word_max - 1

    x = Fxp(np.array([1.0, -2.0, 3.0]), signed=True, n_word=16, n_frac=0)
    x2 = Fxp(np.array([[1.0, -2.0], [3.0, 4.0]]), signed=True, n_word=16, n_frac=0)

    y = fxp.fxp_max(x, out=Fxp(None, signed=True, n_word=96, n_frac=n_frac))
    assert y() == 3.0

    y = fxp.fxp_min(x, out=Fxp(None, signed=True, n_word=96, n_frac=n_frac))
    assert y() == -2.0

    y = np.sum(x, out=Fxp(None, signed=True, n_word=96, n_frac=n_frac))
    assert y() == 2.0

    y = np.cumsum(x, out=Fxp(np.zeros(3), signed=True, n_word=96, n_frac=n_frac))
    assert np.all(y() == np.array([1.0, -1.0, 2.0]))

    y = fxp.sort(x, out=Fxp(np.zeros(3), signed=True, n_word=96, n_frac=n_frac))
    assert np.all(y() == np.array([-2.0, 1.0, 3.0]))

    y = fxp.transpose(x2, out=Fxp(np.zeros((2, 2)), signed=True, n_word=96, n_frac=n_frac))
    assert np.all(y() == np.array([[1.0, 3.0], [-2.0, 4.0]]))

    y = fxp.diagonal(x2, out=Fxp(np.zeros(2), signed=True, n_word=96, n_frac=n_frac))
    assert np.all(y() == np.array([1.0, 4.0]))

    y = fxp.trace(x2, out=Fxp(None, signed=True, n_word=96, n_frac=n_frac))
    assert y() == 5.0

    y = np.prod(x, out=Fxp(None, signed=True, n_word=96, n_frac=n_frac))
    assert y() == -6.0

    y = np.dot(x, Fxp(np.array([1.0, 1.0, 1.0]), signed=True, n_word=16, n_frac=0), out=Fxp(None, signed=True, n_word=96, n_frac=n_frac))
    assert y() == 2.0
    
