"""
fxpmath

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
SOFTWARE.
"""

#%%
import numpy as np
from . import _n_word_max

#%% 
def array_support(func):
    def iterator(*args, **kwargs):
        if isinstance(args[0], (list, np.ndarray)) and args[0].ndim > 0:
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
    if val < 0:
        val = (1 << nbits) + val
    else:
        val = val % (1 << nbits) 
        if (int(val) & (1 << (nbits - 1))) != 0:
            val = val - (1 << nbits)
    return val

def strbin2int(x, signed=True, n_word=None, return_sizes=False):

    x = x.split('b')[-1]        # remove 0b at the begining
    x = x.replace(' ', '')      # remove spacing

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
        val = int(x[1:], 2)
        if x[0] == '1':
            val = -1*( (1 << (n_word - 1)) - val)
    else:
        val = int(x, 2)

    if return_sizes:
        return val, signed, n_word
    else:
        return val

def strbin2float(x, signed=True, n_word=None, n_frac=None, return_sizes=False):
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

def strhex2int(x, signed=True, n_word=None, return_sizes=False):
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
                val, signed, n_word, n_frac =  strbin2float(x, signed, n_word, n_frac, return_sizes=True)
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

def binary_repr(x, n_word=None, n_frac=None):
    if n_frac is None:
        val = np.binary_repr(x, width=n_word)
    else:
        val = insert_frac_point(np.binary_repr(x, width=n_word), n_frac=n_frac)
    return val

def hex_repr(x, n_word=None, padding=None):
    if n_word is not None:
        val = '0x{0:0{1}X}'.format(x, int(np.ceil(n_word/4)))
    elif padding is not None:
        val = '0x{0:0{1}X}'.format(x, padding)
    else:
        val = hex(x)
        val = '0x'+val[2:].upper()
    return val  

def base_repr(x, n_word=None, base=2, n_frac=None):
    if n_frac is None:
        val = np.base_repr(x, base=base)
    elif base == 2:
        val = insert_frac_point(np.base_repr(x, base=base), n_frac=n_frac)
    return val

def bits_len(x, signed=None):
    if signed is None and x < 0:
        signed = True
    elif signed is None:
        signed = False
    elif not signed and x < 0:
        raise ValueError('negative value and unsigned type are incompatible!')

    n_bits = max( np.ceil(np.log2(np.abs(int(x)+0.5))).astype(int), 0) + signed
    return n_bits

def min_pow2(x, n_frac=0):
    _pow = 1
    x = np.array(x)

    if np.any(x != 0):
        while not np.any(x % 2**_pow):
            _pow += 1
        _pow -= n_frac + 1 
    else:
        _pow = None
    
    return _pow
    

@array_support
def binary_invert(x, n_word=None):
    if n_word is None:
        n_word = bits_len(x)
    return int((1 << n_word) - 1 - x)

@array_support
def binary_and(x, y, n_word=None):
    xm = int(x) % (1 << n_word)
    ym = int(y) % (1 << n_word)
    z = xm & ym
    return z

@array_support
def binary_or(x, y, n_word=None):
    xm = int(x) % (1 << n_word)
    ym = int(y) % (1 << n_word)
    z = xm | ym
    return z

@array_support
def binary_xor(x, y, n_word=None):
    xm = int(x) % (1 << n_word)
    ym = int(y) % (1 << n_word)
    z = xm ^ ym
    return z

@np.vectorize
def clip(x, val_min, val_max):
    x_clipped = np.array(max(val_min, min(val_max, x)))
    return x_clipped

@np.vectorize
def int_clip(x, val_min, val_max):
    x_clipped = np.array(max(val_min, min(val_max, int(x))))
    return x_clipped

def wrap(x, signed, n_word):
    if n_word >= _n_word_max:
        dtype = object
    else:
        dtype = int

    m = (1 << n_word)
    if signed: 
        x = np.array(x).astype(dtype) & (m - 1)
        x = np.asarray(x).astype(dtype)
        x = np.where(x < (1 << (n_word-1)), x, x | (-m)) 
    else: 
        x = np.array(x).astype(dtype) & (m - 1) 
        x = np.asarray(x).astype(dtype)
    return x

def get_sizes_from_dtype(dtype):
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