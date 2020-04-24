from .objects import Fxp

def fxp_like(x, val=None):
    return Fxp(val=val, signed=x.signed, n_word=x.n_word, n_frac=x.n_frac)