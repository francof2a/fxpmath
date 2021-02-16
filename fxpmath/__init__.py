__version__ = '0.4.0-rc.2'

import sys
import os

_INFO_PRINT_ENABLE = False

# check if __array_function__ methods is enabled in numpy
# os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "1"
_NUMPY_EXPERIMENTAL_ARRAY_FUNCTION_AUTOENABLE = True
if "NUMPY_EXPERIMENTAL_ARRAY_FUNCTION" in os.environ.keys():
    _NUMPY_EXPERIMENTAL_ARRAY_FUNCTION =  int(os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"])
    if _NUMPY_EXPERIMENTAL_ARRAY_FUNCTION == "0" and _NUMPY_EXPERIMENTAL_ARRAY_FUNCTION_AUTOENABLE:
        if _INFO_PRINT_ENABLE: print('info: auto enabling NUMPY_EXPERIMENTAL_ARRAY_FUNCTION')
        os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "1"
else:
    if _NUMPY_EXPERIMENTAL_ARRAY_FUNCTION_AUTOENABLE:
        if _INFO_PRINT_ENABLE: print('info: creating and enabling NUMPY_EXPERIMENTAL_ARRAY_FUNCTION')
        os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "1"
    elif _INFO_PRINT_ENABLE:
        print('info: environment variable NUMPY_EXPERIMENTAL_ARRAY_FUNCTION not defined')



# max size constant
try:
    import numpy as np
    __maxsize__ = sys.maxsize
    _n_word_max = int(np.log2(__maxsize__)) + 1
except:
    # print("Max size for integer couldn't be found for this computer. n_word max = 64 bits.")
    _n_word_max = 64

try:
    _max_error = 1 / (1 << (_n_word_max - 1))
except:
    _max_error = 1 / 2**63


from . import objects
from . import functions
from .objects import (
    Fxp,
    Config
)

from .functions import (
    fxp_like,
    fxp_sum,
    fxp_max,
    fxp_min,
    add,
    sub,
    mul,
    truediv,
    floordiv,
    mod,
    pow,
    sum,
    cumsum,
    cumprod,
    sort,
    conjugate,
    transpose,
    clip,
    diagonal,
    trace,
    prod,
    dot,
    nonzero
)
