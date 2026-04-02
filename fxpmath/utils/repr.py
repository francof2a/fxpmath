"""String/representation helpers for fixed-point values."""

import numpy as np

from .common import array_support
@array_support
def twos_complement_repr(val, nbits):
    """Convert values to signed two's-complement representation.
    
    Parameters
    ---
    val : int or numpy.ndarray
        Input integer value(s).
    nbits : int
        Bit width used for two's-complement interpretation.
    
    Returns
    ---
    int or numpy.ndarray
        Value represented in signed two's-complement form."""
    if val < 0:
        val = (1 << nbits) + val
    else:
        val = val % (1 << nbits) 
        if (int(val) & (1 << (nbits - 1))) != 0:
            val = val - (1 << nbits)
    return val


def insert_frac_point(x_bin, n_frac):
    """Insert a binary point into a bit-string representation at the requested fractional position.
    
    Parameters
    ---
    x_bin : str
        Binary digit string without spacing normalization issues.
    n_frac : int
        Number of digits placed after the inserted radix point.
    
    Returns
    ---
    str
        Input bit string with radix point inserted at requested fractional position."""
    if n_frac is not None:
        x_bin = x_bin.replace('0b', '')
        # sign
        if x_bin[0] == '-' or x_bin[0] == '+':
            sign_symbol = x_bin[0]
            x_bin = x_bin[1:]
        else:
            sign_symbol = ''

        if len(x_bin) > n_frac > 0:
            x_bin = x_bin[0:-n_frac] + '.' + x_bin[-n_frac:]
        elif n_frac == 0:
            x_bin = x_bin + '.'
        elif n_frac < 0:
            x_bin = x_bin + '#'*(-n_frac) + '.'
        elif n_frac == len(x_bin):
            x_bin = '.' + x_bin
        elif n_frac > len(x_bin):
            x_bin = '.' + '0'*(n_frac - len(x_bin)) + x_bin

        x_bin = sign_symbol + x_bin
    
    return x_bin

@array_support
def binary_repr(x, n_word=None, n_frac=None, prefix=None):
    """Return a binary representation string.
    
    Parameters
    ---
    x : int
        Integer value to format in binary.
    n_word : int, optional
        Minimum word length used for output padding and wrapping.
    n_frac : int, optional
        Fractional-bit count used to insert a radix point.
    prefix : str or None, optional
        Prefix prepended to each formatted output string.
    
    Returns
    ---
    str
        Binary representation string."""
    if n_frac is None:
        val = np.binary_repr(int(x), width=n_word)
    else:
        val = insert_frac_point(np.binary_repr(x, width=n_word), n_frac=n_frac)

    if prefix is not None:
        val = add_binary_prefix(val, prefix=prefix)
    return val

@array_support
def hex_repr(x, n_word=None, padding=None, base=10, prefix='0x'):
    """Return a hexadecimal representation string.
    
    Parameters
    ---
    x : int
        Integer value to format in hexadecimal.
    n_word : int, optional
        Word length used to determine hexadecimal digit padding.
    padding : int, optional
        Minimum number of hexadecimal digits in the output.
    base : int, optional
        Numeric base, kept for compatibility with shared formatter logic.
    prefix : str or None, optional
        Prefix prepended to output (for example `0x`).
    
    Returns
    ---
    str
        Hexadecimal representation string."""
    if base == 2:
        x = int(x, 2)
    elif base == 10:
        pass
    else:
        raise ValueError('base {base} for input value is not supported!')

    if n_word is not None:
        val = prefix + '{0:0{1}X}'.format(x, int(np.ceil(n_word/4)))
    elif padding is not None:
        val = prefix + '{0:0{1}X}'.format(x, padding)
    else:
        val = hex(x)
        val = prefix + val[2:].upper()
    return val  

@array_support
def base_repr(x, n_word=None, base=2, n_frac=None):
    """Return a base-N representation string.
    
    Parameters
    ---
    x : int
        Integer value to format.
    n_word : int, optional
        Word length used when zero-padding binary/hex outputs.
    base : int, optional
        Output base (2..36).
    n_frac : int, optional
        Fractional-bit count used for radix-point insertion.
    
    Returns
    ---
    str
        Base-N representation string."""
    if n_frac is None:
        val = np.base_repr(x, base=base)
    elif base == 2:
        val = insert_frac_point(np.base_repr(x, base=base), n_frac=n_frac)
    else:
        val = np.base_repr(x, base=base)
    return val

@array_support
def add_binary_prefix(x, prefix='0b'):
    """Normalize binary strings so they include a `0b` prefix for each component.
    
    Parameters
    ---
    x : str, list[str], or numpy.ndarray
        Binary string(s) that may or may not already include a `0b` prefix.
    prefix : str or None, optional
        Prefix token inserted ahead of each binary token.
    
    Returns
    ---
    str or numpy.ndarray
        Input string(s) with normalized binary prefixes."""
    if isinstance(x, np.ndarray) and x.ndim == 0:
        x = x.item()

    if isinstance(x, str):
        # convert to easy format
        x = x.lower().replace(' ', '').replace('i', 'j').replace('0b', '').replace('b', '')

        if ('+' in x or '-' in x) and 'j' in x:
            # complex format
            x = prefix + x.replace('+', '+' + prefix).replace('-', '-' + prefix)
        else:
            x = prefix + x
        
        # check valid characters
        invalid_chars = set(x.replace(prefix, '')) - {'0', '1', '.', 'j', '+', '-'}
        if len(invalid_chars) > 0:
            raise ValueError(f"Binary string has invalid characters: {invalid_chars}")
    else:
        raise ValueError("Binary value must be a string!")
    
    return x

def complex_repr(r, i):
    """Return a formatted complex representation string.
    
    Parameters
    ---
    r : str or array_like
        Real-part string(s) used to build complex literals.
    i : str or array_like
        Imaginary-part string(s) used to build complex literals.
    
    Returns
    ---
    str or numpy.ndarray
        Complex-number string assembled from real and imaginary parts."""
    r = np.asarray(r)
    i = np.asarray(i)

    assert r.shape == i.shape

    c = np.empty(r.shape, dtype=object)

    if r.dtype.type is np.str_ and i.dtype.type is np.str_:
        for idx in np.ndindex(r.shape):
            imag_sign_symbol = '' if ('-' in str(i[idx]) or '+' in str(i[idx])) else '+'
            c[idx] = str(r[idx]) + imag_sign_symbol + str(i[idx]) + 'j'
    else:
        raise ValueError('parameters must be a list of array of strings!')
    
    # return single element is array has one value
    if c.size == 0:
        c = c.item(0)
    return c
