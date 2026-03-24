import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import warnings

from fxpmath.objects import Fxp
from fxpmath import functions


def test_reshape_returns_new_object_and_preserves_source_shape():
    x = Fxp([1, 2, 3, 4], signed=True, n_word=8, n_frac=0)

    y = x.reshape((2, 2))

    assert y is not x
    assert x.shape == (4,)
    assert y.shape == (2, 2)
    assert np.all(x() == np.array([1, 2, 3, 4]))
    assert np.all(y() == np.array([[1, 2], [3, 4]]))


def test_reshape_inplace_mutates_and_returns_self():
    x = Fxp([1, 2, 3, 4], signed=True, n_word=8, n_frac=0)

    out = x.reshape_inplace((2, 2))

    assert out is x
    assert x.shape == (2, 2)
    assert x.ndim == 2
    assert x.size == 4


def test_shape_setter_warns_and_reshapes_inplace():
    x = Fxp([1, 2, 3, 4], signed=True, n_word=8, n_frac=0)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', DeprecationWarning)
        x.shape = (2, 2)

    assert x.shape == (2, 2)
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_shape_setter_raises_for_invalid_size():
    x = Fxp([1, 2, 3, 4], signed=True, n_word=8, n_frac=0)

    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always', DeprecationWarning)
        try:
            x.shape = (3, 2)
            assert False
        except ValueError:
            pass


def test_functions_reshape_accepts_shape_and_newshape_alias():
    x = Fxp([1, 2, 3, 4], signed=True, n_word=8, n_frac=0)

    y_shape = functions.reshape(x, shape=(2, 2))
    y_newshape = functions.reshape(x, newshape=(2, 2))

    assert y_shape.shape == (2, 2)
    assert y_newshape.shape == (2, 2)
    assert np.all(y_shape() == y_newshape())


def test_reshape_preserves_core_config_and_dtype_fields():
    x = Fxp([1.0, 2.0, 3.0, 4.0], signed=True, n_word=16, n_frac=4, overflow='wrap', rounding='floor')

    y = x.reshape((2, 2))

    assert y.signed == x.signed
    assert y.n_word == x.n_word
    assert y.n_frac == x.n_frac
    assert y.overflow == x.overflow
    assert y.rounding == x.rounding


def test_size_shape_ndim_stay_consistent_across_ops():
    x = Fxp(np.arange(12), signed=True, n_word=16, n_frac=0)

    y = x.reshape((3, 4))
    z = y.reshape((2, 6))

    assert x.size == 12 and x.shape == (12,) and x.ndim == 1
    assert y.size == 12 and y.shape == (3, 4) and y.ndim == 2
    assert z.size == 12 and z.shape == (2, 6) and z.ndim == 2
