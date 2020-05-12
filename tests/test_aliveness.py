import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import fxpmath as fxp
from fxpmath import Fxp

def test_aliveness():
    x = Fxp(0.0, signed=True, n_word=8, n_frac=2)
    assert x(4.75) == 4.75