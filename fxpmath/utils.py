"""fxpmath

---

A python library for fractional fixed-point arithmetic.

---

This software is provided under MIT License:

MIT License

Copyright (c) 2020 Franco, francof2a

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""

#%%
import numpy as np
import warnings
from . import _n_word_max

_mixed_complex_bitwise_warned = False

class ComplexBitwiseOperationWarning(UserWarning):
    """Warning emitted once when mixing complex and non-complex bitwise operands."""

#%% 
def array_support(func):
    """Decorate scalar helpers to operate element-wise on arrays.
    
    Parameters
    ---
    func : Callable
        Scalar helper function to decorate with array support.
    
    Returns
    ---
    Callable
        Decorator wrapper that adds recursive array handling to a scalar helper."""
    def iterator(*args, **kwargs):
        """Recursively apply the wrapped scalar helper element-wise over array-like inputs.
        
        Parameters
        ---
        *args : tuple
            Extra positional arguments forwarded to the wrapped callable.
        **kwargs : dict
            Extra keyword arguments forwarded to the underlying NumPy operation.
        
        Returns
        ---
        object
            Computed value."""
        if isinstance(args[0], (list, np.ndarray)) and np.asarray(args[0]).ndim > 0:
            vals = []
            for v in args[0]:
                vals.append(iterator(v, *args[1:], **kwargs))

            if isinstance(args[0], np.ndarray):
                vals = np.array(vals)
            return vals
        else:
            return func(*args, **kwargs)
    return iterator

#%%
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

def bits_len(x, signed=None):
    """Return the minimum number of bits required to represent the integer part of a value.
    
    Parameters
    ---
    x : int
        Integer value whose required bit width is measured.
    signed : bool, optional
        When `True`, include one sign bit in the returned length.
    
    Returns
    ---
    int
        Minimum bit width needed to represent the value."""
    if signed is None and x < 0:
        signed = True
    elif signed is None:
        signed = False
    elif not signed and x < 0:
        raise ValueError('negative value and unsigned type are incompatible!')

    n_bits = max( np.ceil(np.log2(np.abs(int(x)+0.5))).astype(int), 0) + signed
    return n_bits

def min_pow2(x, n_frac=0):
    """Return the smallest exponent `p` such that `2**p >= value` in magnitude.
    
    Parameters
    ---
    x : float
        Positive value used to compute floor(log2(x)).
    n_frac : int, optional
        Reference fractional scaling, used when values are already quantized.
    
    Returns
    ---
    int
        Greatest integer exponent `n` such that `2**n <= x`."""
    _pow = 1
    x = np.array(x)

    if np.any(x != 0):
        while not np.any(x % 2**_pow):
            _pow += 1
        _pow -= n_frac + 1 
    else:
        _pow = None
    
    return _pow
    

def _bitwise_infer_n_word(*vals):
    """Infer bit width for bitwise ops when n_word is omitted."""
    inferred = 0
    for val in vals:
        arr = np.asarray(val)
        if arr.ndim == 0:
            inferred = max(inferred, bits_len(arr.item()))
        elif arr.size > 0:
            for item in arr.flat:
                inferred = max(inferred, bits_len(item))
    return inferred


def _bitwise_binary_apply(x, y, n_word, pyop):
    """Apply a binary bitwise op with NumPy broadcasting semantics."""
    if n_word is None:
        n_word = _bitwise_infer_n_word(x, y)
    n_word = int(n_word)
    mod = 1 << n_word

    xa = np.asarray(x)
    ya = np.asarray(y)
    x_b, y_b = np.broadcast_arrays(xa, ya)

    op = np.frompyfunc(lambda a, b: pyop(int(a) % mod, int(b) % mod), 2, 1)
    z = np.asarray(op(x_b, y_b), dtype=object)

    if xa.ndim == 0 and ya.ndim == 0:
        return int(z.item())
    return z


def _bitwise_unary_apply(x, n_word, pyop):
    """Apply a unary bitwise op over scalar or array inputs."""
    if n_word is None:
        n_word = _bitwise_infer_n_word(x)
    n_word = int(n_word)
    mod = 1 << n_word

    xa = np.asarray(x)
    op = np.frompyfunc(lambda a: pyop(int(a) % mod), 1, 1)
    z = np.asarray(op(xa), dtype=object)

    if xa.ndim == 0:
        return int(z.item())
    return z


def binary_invert(x, n_word=None):
    """Apply bitwise NOT with broadcasting-friendly behavior."""
    if n_word is None:
        n_word = _bitwise_infer_n_word(x)
    n_word = int(n_word)
    mod_minus_one = (1 << n_word) - 1
    return _bitwise_unary_apply(x, n_word=n_word, pyop=lambda a: mod_minus_one - a)


def binary_and(x, y, n_word=None):
    """Apply bitwise AND with NumPy broadcasting semantics."""
    return _bitwise_binary_apply(x, y, n_word=n_word, pyop=lambda a, b: a & b)


def binary_or(x, y, n_word=None):
    """Apply bitwise OR with NumPy broadcasting semantics."""
    return _bitwise_binary_apply(x, y, n_word=n_word, pyop=lambda a, b: a | b)


def binary_xor(x, y, n_word=None):
    """Apply bitwise XOR with NumPy broadcasting semantics."""
    return _bitwise_binary_apply(x, y, n_word=n_word, pyop=lambda a, b: a ^ b)

def is_complex_data(x):
    """Return True when an operand contains complex values."""
    return isinstance(x, complex) or np.iscomplexobj(x)


def bitwise_result_dtype(base_vdtype, force_complex=False):
    """Preserve base vdtype unless operation requires complex output."""
    if force_complex:
        return complex
    return base_vdtype


def reset_mixed_complex_bitwise_warning_state():
    """Reset the one-time mixed-complex bitwise warning flag."""
    global _mixed_complex_bitwise_warned
    _mixed_complex_bitwise_warned = False


def warn_mixed_complex_bitwise_once(stacklevel=2):
    """Warn once when bitwise ops mix complex and non-complex operands."""
    global _mixed_complex_bitwise_warned
    if not _mixed_complex_bitwise_warned:
        warnings.warn(
            'Bitwise operation mixed complex and non-complex operands; applying the real operand to both real and imaginary parts.',
            ComplexBitwiseOperationWarning,
            stacklevel=stacklevel,
        )
        _mixed_complex_bitwise_warned = True


def twos_complement_componentwise(val, nbits):
    """Apply two's-complement conversion to real/imag parts independently."""
    if is_complex_data(val):
        real_val = twos_complement_repr(np.real(val), nbits=nbits)
        imag_val = twos_complement_repr(np.imag(val), nbits=nbits)
        return real_val + 1j * imag_val
    return twos_complement_repr(val, nbits=nbits)


def binary_invert_componentwise(x, n_word=None):
    """Apply bitwise invert, including component-wise handling for complex values."""
    if is_complex_data(x):
        real_val = binary_invert(np.real(x), n_word=n_word)
        imag_val = binary_invert(np.imag(x), n_word=n_word)
        return real_val + 1j * imag_val, True

    return binary_invert(x, n_word=n_word), False


def binary_op_componentwise(x, y, op, n_word=None, warn_mixed=True, warning_stacklevel=2):
    """Apply a binary bitwise op with complex component-wise semantics."""
    x_is_complex = is_complex_data(x)
    y_is_complex = is_complex_data(y)
    force_complex = x_is_complex or y_is_complex

    if not force_complex:
        return op(x, y, n_word=n_word), False

    if x_is_complex and y_is_complex:
        left_real = np.real(x)
        left_imag = np.imag(x)
        right_real = np.real(y)
        right_imag = np.imag(y)
    elif x_is_complex:
        if warn_mixed:
            warn_mixed_complex_bitwise_once(stacklevel=warning_stacklevel)
        left_real = np.real(x)
        left_imag = np.imag(x)
        right_real = y
        right_imag = y
    else:
        if warn_mixed:
            warn_mixed_complex_bitwise_once(stacklevel=warning_stacklevel)
        left_real = x
        left_imag = x
        right_real = np.real(y)
        right_imag = np.imag(y)

    out_real = op(left_real, right_real, n_word=n_word)
    out_imag = op(left_imag, right_imag, n_word=n_word)
    return out_real + 1j * out_imag, True

@np.vectorize
def clip(x, val_min, val_max):
    """Clip fixed-point values to minimum and maximum bounds.
    
    Parameters
    ---
    x : scalar or numpy.ndarray
        Input value(s) to clip.
    val_min : scalar
        Lower clipping bound.
    val_max : scalar
        Upper clipping bound.
    
    Returns
    ---
    scalar or numpy.ndarray
        Clipped value(s) within [`val_min`, `val_max`]."""
    x_clipped = np.array(max(val_min, min(val_max, x)))
    return x_clipped

@np.vectorize
def int_clip(x, val_min, val_max):
    """Clip integer values between minimum and maximum limits.
    
    Parameters
    ---
    x : int or numpy.ndarray
        Integer value(s) to clip.
    val_min : int
        Lower clipping bound.
    val_max : int
        Upper clipping bound.
    
    Returns
    ---
    int or numpy.ndarray
        Integer-clipped value(s) within bounds."""
    x_clipped = np.array(max(val_min, min(val_max, int(x))))
    return x_clipped

def wrap(x, signed, n_word):

    """Wrap integers into the representable range using modular arithmetic.
    
    Parameters
    ---
    x : int or numpy.ndarray
        Raw integer value(s) to wrap to fixed-point range.
    signed : bool
        Whether to wrap to signed or unsigned range.
    n_word : int
        Word length that defines wrapping period.
    
    Returns
    ---
    int or numpy.ndarray
        Wrapped integer value(s) constrained to fixed-point range."""
    m = (1 << n_word)
    if n_word >= _n_word_max:
        dtype = object
        x = int_array(x).astype(dtype) & (m - 1)
    else:
        dtype = int
        x = np.array(x).astype(dtype) & (m - 1) 

    x = np.asarray(x).astype(dtype)

    if signed: 
        x = np.where(x < (1 << (n_word-1)), x, x | (-m))
        
    return x

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


# def int_array(x):
#     x = np.array(x) 
#     int_vectorized = np.vectorize(int)

#     if x.dtype != complex:
#         y = np.array(int_vectorized(x))
#     else:
#         y = np.array(int_vectorized(x.real) + 1j*int_vectorized(x.imag))
    
#     return y

def int_array(x):
    """Convert inputs to integer ndarrays using safe dtypes for large values.
    
    Parameters
    ---
    x : scalar or array_like
        Input numeric values to cast to integer storage form.
    
    Returns
    ---
    int or numpy.ndarray
        Integer-cast scalar or array."""
    if not isinstance(x, np.ndarray):
        x = np.array(x)

    if x.dtype != complex:
        x = np.array(list(map(int, x.flatten()))).reshape(x.shape)
    else:
        x_real = np.vectorize(lambda v: v.real)(x)
        x_imag = np.vectorize(lambda v: v.imag)(x)
        x_real = np.array(list(map(int, x_real.flatten()))).reshape(x_real.shape)
        x_imag = np.array(list(map(int, x_imag.flatten()))).reshape(x_imag.shape)
        x = np.array(x_real + 1j*x_imag)
    return x