from .objects import Fxp

def fxp_like(x, val=None):
    y = x.copy()
    return y(val)