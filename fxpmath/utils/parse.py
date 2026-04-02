"""Parsing helpers for binary/hex/string fixed-point inputs."""

import numpy as np

from .repr import add_binary_prefix
def strbin2int(x, signed=True, n_word=None, return_sizes=False):

    """Convert binary string input into integer values.
    
    Parameters
    ---
    x : str
        Binary string to parse (prefix/spaces accepted).
    signed : bool, optional
        Whether to interpret the value using signed two's-complement rules.
    n_word : int, optional
        Expected word length. When omitted, it is inferred from the string length.
    return_sizes : bool, optional
        When `True`, return parsed value together with inferred size metadata.
    
    Returns
    ---
    int or tuple
        Parsed integer value, optionally with inferred sizes."""
    x = x.replace('0b', 'b').replace('b', '')       # remove 0b at the begining
    x = x.replace(' ', '').replace('+', '')         # remove spacing and +

    # get original sign of number
    sign = -1 if x[0] == '-' else 1
    x = x.replace('-', '')

    if n_word is None:
        n_word = len(x)
    elif len(x) < n_word:
        if signed:
            x = x[0]*(n_word - len(x)) + x      # expand original binary with sign bit
        else:
            x = '0'*(n_word - len(x)) + x       # expand original binary with zeros
    elif len(x) > n_word:
        raise ValueError('binary val has more bits ({}) than word ({})!'.format(len(x), n_word))
    
    if signed:
        if len(x) < 2:
            raise('Signed binary with no enough amount of bits!')
        
        val = int(x[1:], 2)
        if x[0] == '1':
            val = -1*( (1 << (n_word - 1)) - val)
        
        if sign == -1:
            print('Warning: you are using a negative sign (-) with an already binary signed. The value conversion could be wrong!')
    else:
        val = int(x, 2)

    # set same original sign
    val = sign * val

    if return_sizes:
        return val, signed, n_word
    else:
        return val

def strbin2float(x, signed=True, n_word=None, n_frac=None, return_sizes=False):
    """Convert binary string input into floating-point values.
    
    Parameters
    ---
    x : str
        Binary fixed-point string to parse.
    signed : bool, optional
        Whether to interpret the value using signed two's-complement rules.
    n_word : int, optional
        Expected word length.
    n_frac : int, optional
        Number of fractional bits; inferred from radix point when omitted.
    return_sizes : bool, optional
        When `True`, return parsed value together with inferred size metadata.
    
    Returns
    ---
    float or tuple
        Parsed fixed-point value, optionally with inferred sizes."""
    if n_frac is None:
        if '.' in x:
            point_idx = x.find('.')
            n_frac = len(x) - point_idx - 1     # number of bits after dot
        else:
            n_frac = 0
    else:
        if '.' in x:
            point_idx = x.find('.')
            x = x + '0'*(n_frac - (len(x) - point_idx - 1))     # complete with zeros the frac part

    x = x.replace('.', '')
    val, signed, n_word = strbin2int(x, signed, n_word, return_sizes=True)
    val /= (2**n_frac) 
    
    if return_sizes:
        return val, signed, n_word, n_frac
    else:
        return val

def strbin2complex(x, signed=True, n_word=None, n_frac=None, return_sizes=False):
    """Convert binary string input into complex values.
    
    Parameters
    ---
    x : str
        Complex binary string in forms like `0b01+0b10j`.
    signed : bool, optional
        Whether to interpret components with signed two's-complement rules.
    n_word : int, optional
        Expected word length for each component.
    n_frac : int, optional
        Number of fractional bits for each component.
    return_sizes : bool, optional
        When `True`, return parsed value together with inferred size metadata.
    
    Returns
    ---
    complex or tuple
        Parsed complex value, optionally with inferred sizes."""
    x = x.replace(' ', '').replace('+', '|').replace('-', '|-').split('|')

    if len(x) == 1  and isinstance(x[0], str) and 'j' in x[0]:
        # imaginary number
        val, signed, n_word, n_frac = strbin2float(x[0].replace('j', ''), signed, n_word, n_frac, return_sizes=True)
        val = 1j*val
    elif len(x) == 1 and isinstance(x[0], str) and not 'j' in x[0]:
        # real number
        val, signed, n_word, n_frac = strbin2float(x[0], signed, n_word, n_frac, return_sizes=True)
        val = val + 1j*0
    elif len(x) == 2 and isinstance(x, list) and not 'j' in x[0] and 'j' in x[1]:
        # complex
        val_real, signed_real, n_word_real, n_frac_real = strbin2float(x[0], signed, n_word, n_frac, return_sizes=True)
        val_imag, signed_imag, n_word_imag, n_frac_imag = strbin2float(x[1].replace('j', ''), signed, n_word, n_frac, return_sizes=True)
        val = val_real + 1j*val_imag

        signed = signed_real or signed_imag
        n_word = max(n_word_real, n_word_imag)
        n_frac = max(n_frac_real, n_frac_imag)
    else:
        raise ValueError(f"Wrong complex format of binary string!")
    
    if return_sizes:
        return val, signed, n_word, n_frac
    else:
        return val


def strhex2int(x, signed=True, n_word=None, return_sizes=False):
    """Convert hexadecimal string input into integer values.
    
    Parameters
    ---
    x : str
        Hexadecimal string to parse.
    signed : bool, optional
        Whether to interpret the value using signed two's-complement rules.
    n_word : int, optional
        Expected word length in bits.
    return_sizes : bool, optional
        When `True`, return parsed value together with inferred size metadata.
    
    Returns
    ---
    int or tuple
        Parsed integer value from hexadecimal input."""
    x = x.replace('0x', '')
    if n_word is None:
        n_word = len(x)*4

    x_bin = bin(int(x, 16))

    if len(x_bin[2:]) < n_word:
        x_bin = '0b' + '0'*(n_word - len(x_bin[2:])) + x_bin[2:]

    val = strbin2int(x_bin, signed, n_word)

    if return_sizes:
        return val, signed, n_word
    else:
        return val

def strhex2float(x, signed=True, n_word=None, n_frac=None, return_sizes=False):
    """Convert hexadecimal string input into floating-point values.
    
    Parameters
    ---
    x : str
        Hexadecimal fixed-point string to parse.
    signed : bool, optional
        Whether to interpret the value using signed two's-complement rules.
    n_word : int, optional
        Expected word length in bits.
    n_frac : int, optional
        Number of fractional bits used to scale the parsed value.
    return_sizes : bool, optional
        When `True`, return parsed value together with inferred size metadata.
    
    Returns
    ---
    float or tuple
        Parsed fixed-point value from hexadecimal input."""
    x = x.replace('0x', '')
    if n_word is None:
        n_word = len(x)*4

    x_bin = bin(int(x, 16))
    
    if len(x_bin[2:]) < n_word:
        x_bin = '0b' + '0'*(n_word - len(x_bin[2:])) + x_bin[2:]

    val, signed, n_word, n_frac = strbin2float(x_bin, signed, n_word, n_frac, return_sizes=True)

    if return_sizes:
        return val, signed, n_word, n_frac
    else:
        return val

def str2num(x, signed=True, n_word=None, n_frac=None, base=10, return_sizes=False):
    """Parse string input into numeric values.
    
    Parameters
    ---
    x : str, list, numpy.ndarray, or object
        Input literal(s) to convert to numeric values.
    signed : bool, optional
        Signed interpretation used for binary/hex literals.
    n_word : int, optional
        Expected word length for binary/hex literals.
    n_frac : int, optional
        Fractional-bit width for fixed-point literals.
    base : int, optional
        Explicit integer base used for generic string conversion.
    return_sizes : bool, optional
        When `True`, include inferred sizing metadata in results.
    
    Returns
    ---
    number, complex, list, numpy.ndarray, or None
        Converted numeric value(s) preserving container shape when possible."""
    if isinstance(x, (list, tuple)):
        _signed_max = False
        _n_word_max = None
        _n_frac_max = None

        for idx, v in enumerate(x):
            x[idx], _signed, _n_word, _n_frac = str2num(v, signed, n_word, n_frac, base, return_sizes=True)

            _signed = _signed_max or _signed
            if _n_word is not None:
                _n_word_max = _n_word if _n_word_max is None else max(_n_word_max, _n_word)
            if _n_frac is not None:
                _n_frac_max = _n_frac if _n_frac_max is None else max(_n_frac_max, _n_frac)

        val = x
        signed = signed or _signed
        n_word = _n_word_max if n_word is None else n_word
        n_frac = _n_frac_max if n_frac is None else n_frac

    elif isinstance(x, str):
        x = x.replace('h', 'x')     # for hex numbers: h -> x

        if base == 2 or 'b' in x[:2]:
            # binary
            if '.' in x or (n_frac is not None and n_frac > 0):
                # fractional binary
                if 'j' in x:
                    val, signed, n_word, n_frac =  strbin2complex(x, signed, n_word, n_frac, return_sizes=True)
                else:
                    val, signed, n_word, n_frac =  strbin2float(x, signed, n_word, n_frac, return_sizes=True)
            else:
                # integer binary
                if 'j' in x:
                    val, signed, n_word = strbin2complex(x, signed, n_word, return_sizes=True)
                else:
                    val, signed, n_word = strbin2int(x, signed, n_word, return_sizes=True)
                n_frac = 0
            
        elif base == 16 or 'x' in x[:2]:
            if n_frac is not None and n_frac > 0:
                val, signed, n_word, n_frac = strhex2float(x, signed, n_word, n_frac, return_sizes=True)
            else:
                val, signed, n_word = strhex2int(x, signed, n_word, return_sizes=True)
                n_frac = 0

        elif base == 10:
            if '.' in x or (n_frac is not None and n_frac > 0):
                val = float(x)
            else:
                val = int(x)

        elif base is not None:
            val = int(x, base)

        else:
            raise ValueError('string format not supported for conversion or its base is ambiguous!')
    else:
        val = x
    
    if return_sizes:
        return val, signed, n_word, n_frac
    else:
        return val


def get_sizes_from_dtype(dtype):
    """Parse dtype notation and return fixed-point size information.
    
    Parameters
    ---
    dtype : str or None, optional
        Fixed-point dtype string used for result construction.
    
    Returns
    ---
    tuple[bool, int, int, int]
        Parsed `(signed, n_word, n_int, n_frac)` tuple extracted from dtype."""
    if isinstance(dtype, str):
        head, props = dtype.split('-')
        if head == 'fxp':
            # sign
            if props[0] == 's':
                signed = True
            elif props[0] == 'u':
                signed = False
            else:
                raise ValueError('dtype sign specifier should be `s` or `u`')

            # sizes
            if '-' in props:
                props, _ = props.split('-')

            n_word, n_frac = props[1:].split('/')
            n_word = int(n_word)
            n_frac = int(n_frac)
        else:
            raise ValueError('dtype str format must be fxp-<sign><n_word>/<n_frac>-<complex>')
    else:
        raise ValueError('dtype must be a str!')

    return signed, n_word, n_frac
