import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import fxpmath as fxp
from fxpmath.objects import Fxp


def test_temp():
    x = Fxp(0.5, True, 8, 7)
    assert x.astype(float) == 0.5
