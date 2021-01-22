__version__ = '0.3.9'

import sys

# max size constant
try:
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
from .objects import *
from .functions import *
