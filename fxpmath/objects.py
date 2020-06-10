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
import copy
from . import utils

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

#%%
class Fxp():
    def __init__(self, val=None, signed=None, n_word=None, n_frac=None, n_int=None, **kwargs):

        # Init all properties in None
        self.dtype = 'fxp' # fxp-<sign><n_word>/<n_frac>-{complex}. i.e.: fxp-s16/15, fxp-u8/1, fxp-s32/24-complex
        # value
        self.vdtype = None # value(s) dtype to return as default
        self.val = None
        self.real = None
        self.imag = None
        # scaling (linear)
        self.scale = None
        self.bias = None
        self.scaled = None
        # format sizes
        self.signed = None
        self.n_word = None
        self.n_frac = None
        self.n_int = None
        # format properties
        self.upper = None
        self.lower = None
        self.precision = None
        #status
        self.status = None
        # behavior
        self.overflow = None
        self.rounding = None
        self.shifting = None
        # size
        max_error = None
        n_word_max = None

        # ---

        # check if init must be a `like` other Fxp
        init_like = kwargs.pop('like', None) 
        if init_like is not None:
            if isinstance(init_like, Fxp):
                self.__dict__ = copy.deepcopy(init_like.__dict__)

        #status (overwrite)
        self.status = {
            'overflow': False,
            'underflow': False,
            'inaccuracy': False}
        # scaling
        if self.scale is None: self.scale = kwargs.pop('scale', 1)
        if self.bias is None: self.bias = kwargs.pop('bias', 0)
        # behavior
        if self.overflow is None: self.overflow = kwargs.pop('overflow', 'saturate')
        if self.rounding is None: self.rounding = kwargs.pop('rounding', 'trunc')
        if self.shifting is None: self.shifting = kwargs.pop('shifting', 'expand')
        # size
        if init_like is None:
            if max_error is None: max_error = kwargs.pop('max_error', _max_error)
            if n_word_max is None: n_word_max = kwargs.pop('n_word_max', _n_word_max)
            self._init_size(val, signed, n_word, n_frac, n_int, max_error=max_error, n_word_max=n_word_max)

        # store the value
        self.set_val(val)


    # methods about size
    def _init_size(self, val=None, signed=None, n_word=None, n_frac=None, n_int=None, max_error=_max_error, n_word_max=_n_word_max):
        # sign by default
        if signed is None:
            self.signed = True
        else:
            self.signed = signed
        
        # n_int defined:
        if n_word is None and n_frac is not None and n_int is not None:
            n_word = n_int + n_frac + (1 if self.signed else 0)
        elif n_frac is None and n_word is not None and n_int is not None:
            n_frac = n_word - n_int - (1 if self.signed else 0)

        # check if I must find the best size for val
        if n_word is None or n_frac is None:
            self.set_best_sizes(val, n_word, n_frac, max_error=max_error, n_word_max=n_word_max)
        else:
            self.resize(self.signed, n_word, n_frac, n_int)


    def resize(self, signed=None, n_word=None, n_frac=None, n_int=None, restore_val=True):
        # n_int defined:
        if n_word is None and n_frac is not None and n_int is not None:
            n_word = n_int + n_frac + (1 if self.signed else 0)
        elif n_frac is None and n_word is not None and n_int is not None:
            n_frac = n_word - n_int - (1 if self.signed else 0)

        # sign
        if signed is not None:
            self.signed = signed
        # word
        if n_word is not None:
            self.n_word = n_word
        # frac
        if n_frac is not None:
            self.n_frac = n_frac
    
        # n_int    
        self.n_int = self.n_word - self.n_frac - (1 if self.signed else 0)

        if self.signed:
            upper_val = (1 << (self.n_word-1)) - 1
            lower_val = -upper_val - 1
        else:
            upper_val =  (1 << self.n_word) - 1
            lower_val = 0 

        if self.vdtype == complex:
            self.upper = (upper_val + 1j * upper_val) / 2.0**self.n_frac
            self.lower = (lower_val + 1j * lower_val) / 2.0**self.n_frac
            self.precision = (1 + 1j * 1) / 2.0**self.n_frac
        else:
            self.upper = upper_val / 2.0**self.n_frac
            self.lower = lower_val / 2.0**self.n_frac
            self.precision = 1 / 2.0**self.n_frac

        # scaling conversion
        if self.scaled:
            self.upper = self.scale * self.upper + self.bias
            self.lower = self.scale * self.lower + self.bias
            self.precision = self.scale * self.precision

        # re store the value
        if restore_val:
            self.set_val(self.get_val())
    
    def set_best_sizes(self, val=None, n_word=None, n_frac=None, max_error=1.0e-6, n_word_max=64):

        if val is None:
            if n_word is None and n_frac is None:
                self.n_word = 16
                self.n_frac = 15
            elif n_frac is None:
                self.n_word = n_word
                self.n_frac = n_word - 1
            elif n_word is None:
                self.n_word = n_frac + 1
                self.n_frac = n_frac
        else:
            if self.signed:
                sign = 1
            else:
                sign = 0

            self.n_word = n_word
            self.n_frac = n_frac
            
            # if val is a str(s), convert to number(s)
            val, signed, n_word, n_frac = self._format_inupt_val(val, return_sizes=True)
            val = np.array([val])

            # check if val is complex, if it is: convert to array of float/int
            if np.iscomplexobj(val):
                val = np.array([val.real, val.imag])

            # define numpy integer type
            if self.signed:
                int_dtype = np.int64
            else:
                int_dtype = np.uint64

            # find fractional parts
            frac_vals = np.abs(np.subtract(val, val.astype(int_dtype))).ravel()

            # n_frac estimation
            if n_frac is None:
                max_n_frac = n_word_max - sign

                n_frac_calcs = []
                for r in frac_vals:
                    e = 1.0
                    n_frac = 0
                    while e > max_error and n_frac <= max_n_frac and r > 0.0:
                        n_frac += 1
                        r_i = r - 0.5**n_frac
                        e = np.abs(r_i)
                        if r_i >= 0.0:
                            r = r_i
                    n_frac_calcs.append(n_frac)
                n_frac = int(max(n_frac_calcs))

            # max raw value (integer) estimation
            n_int = max( np.ceil(np.log2(np.max(np.abs( val*(1 << n_frac) + 0.5 )))).astype(int_dtype) - n_frac, 0)

            # size assignement
            if n_word is None:
                n_frac = min(n_word_max - sign - n_int, n_frac) # n_frac limit according n_word max size
                self.n_frac = int(n_frac)
                self.n_word = int(n_frac + n_int + sign)
            else:
                self.n_word = int(n_word)
                self.n_frac = n_frac = int(min(n_word - sign - n_int, n_frac))
        
        self.n_word = int(min(self.n_word, n_word_max))
        self.resize(restore_val=False)

    # methods about value

    def _format_inupt_val(self, val, return_sizes=False):
        if val is None:
            val = 0
        # if val is a str(s), convert to number(s)
        val, signed, n_word, n_frac = utils.str2num(val, self.signed, self.n_word, self.n_frac, return_sizes=True)
        # convert to (numpy) ndarray
        val = np.array(val)
        # scaling conversion
        self.scaled = False
        if self.scale is not None and self.bias is not None:
            if self.scale != 1 or self.bias != 0:
                self.scaled = True
                val = (val - self.bias) / self.scale

        if return_sizes:
            return val, signed, n_word, n_frac
        else:
            return val

    def set_val(self, val, raw=False, vdtype=None):
        # convert input value to valid format
        val = self._format_inupt_val(val)

        # check if val overflow max int possible
        if val.dtype == 'O':
            raise OverflowError('Integer value too large to convert to C long')

        if self.signed:
            val_max = (1 << (self.n_word-1)) - 1
            val_min = -val_max - 1
            val_dtype = np.int64
        else:
            val_max =  (1 << self.n_word) - 1
            val_min = 0
            val_dtype = np.uint64

        # conversion factor
        if raw:
            conv_factor = 1
        else:
            conv_factor = int(2**self.n_frac)

        # round, saturate and store
        if val.dtype != complex:
            new_val = self._round(val * conv_factor , method=self.rounding)
            new_val = self._overflow_action(new_val, val_min, val_max)
            self.val = new_val.astype(val_dtype)
            self.real = self.get_val()
            self.imag = 0
        else:
            new_val_real = self._round(val.real * conv_factor, method=self.rounding)
            new_val_imag = self._round(val.imag * conv_factor, method=self.rounding)
            new_val_real = self._overflow_action(new_val_real, val_min, val_max).astype(val_dtype)
            new_val_imag = self._overflow_action(new_val_imag, val_min, val_max).astype(val_dtype)
            self.val = new_val = new_val_real + 1j * new_val_imag
            self.real = self.astype(complex).real
            self.imag = self.astype(complex).imag

        # dtype
        self.dtype = 'fxp-{sign}{nword}/{nfrac}{comp}'.format(sign='s' if self.signed else 'u', 
                                                             nword=self.n_word, 
                                                             nfrac=self.n_frac, 
                                                             comp='-complex' if val.dtype == complex else '')

        # vdtype
        if raw:
            if vdtype is not None:
                self.vdtype = vdtype
        else:
            self.vdtype = val.dtype

        # check inaccuray
        if not np.equal(val, new_val/conv_factor).all() :
            self.status['inaccuray'] = True

        return self

    def astype(self, dtype=None):
        if dtype is None:
            dtype = self.vdtype

        if self.val is not None:
            if dtype == float or np.issubdtype(dtype, np.floating):
                val = self.val / 2.0**self.n_frac
            elif dtype == int or dtype == 'uint' or dtype == 'int' or np.issubdtype(dtype, np.integer):
                val = self.val.astype(dtype) // 2**self.n_frac
            elif dtype == complex or np.issubdtype(dtype, np.complexfloating):
                val = (self.val.real + 1j * self.val.imag) / 2.0**self.n_frac
            else:
                val = self.val / 2.0**self.n_frac
        else:
            val = None

        # scaling reconversion
        if val is not None and self.scaled:
            val = val * self.scale + self.bias
        return val

    def get_val(self, dtype=None):
        if dtype is None:
            dtype = self.vdtype
        return self.astype(dtype)

    def equal(self, x):
        if isinstance(x, Fxp):
            x = x()
        self.set_val(x)
        return self

    # behaviors

    def _overflow_action(self, new_val, val_min, val_max):
        if np.any(new_val > val_max):
            self.status['overflow'] = True
        if np.any(new_val < val_min):
            self.status['underflow'] = True
        
        if self.overflow == 'saturate':
            # val = np.clip(new_val, val_min, val_max) # it returns float that cause an error for 64 bits huge integers
            # if new_val.ndim > 0:
            #     val = np.array([max(val_min, min(val_max, v)) for v in new_val])
            # else:
            #     val = np.array(max(val_min, min(val_max, new_val)))
            val = utils.clip(new_val, val_min, val_max)
        elif self.overflow == 'wrap':
            val = utils.wrap(new_val, val_min, val_max, self.signed, self.n_word)
        return val

    def _round(self, val, method='floor'):
        if isinstance(val, int) or val.dtype == int or val.dtype == 'uint':
            rval = val
        elif method == 'around':
            rval = np.around(val)
        elif method == 'floor':
            rval = np.floor(val)
        elif method == 'ceil':
            rval = np.ceil(val)
        elif method == 'fix':
            rval = np.fix(val)
        elif method == 'trunc':
            rval = np.trunc(val)
        elif method is None or method == '':
            rval = val
        else:
            raise ValueError('<{}> rounding method not valid!')
        return rval

    # overloadings

    def __call__(self, val=None):
        if val is None:
            rval = self.get_val()
        else:
            rval = self.set_val(val)
        return rval

    
    # representation
    
    def __repr__(self):
        return str(self.get_val())

    def __str__(self):
        return str(self.get_val())

    
    # math operations
    
    def __neg__(self):
        y = Fxp(-self.get_val(), signed=self.signed, n_word=self.n_word, n_frac=self.n_frac)
        return y

    def __pos__(self):
        y = Fxp(+self.get_val(), signed=self.signed, n_word=self.n_word, n_frac=self.n_frac)
        return y             

    def __add__(self, x):
        if isinstance(x, (int, float, list, np.ndarray)):
            x = Fxp(x)
        
        n_int = max(self.n_int, x.n_int) + 1
        n_frac = max(self.n_frac, x.n_frac)

        y = Fxp(self.get_val() + x.get_val(), signed=self.signed or x.signed, n_int=n_int, n_frac=n_frac)
        return y

    __radd__ = __add__

    __iadd__ = __add__

    def __sub__(self, x):
        if isinstance(x, (int, float, list, np.ndarray)):
            x = Fxp(x)
        
        n_int = max(self.n_int, x.n_int) + 1
        n_frac = max(self.n_frac, x.n_frac)

        y = Fxp(self.get_val() - x.get_val(), signed=self.signed or x.signed, n_int=n_int, n_frac=n_frac)
        return y

    def __rsub__(self, x):
        if isinstance(x, (int, float, list, np.ndarray)):
            x = Fxp(x)
        
        n_int = max(self.n_int, x.n_int) + 1
        n_frac = max(self.n_frac, x.n_frac)

        y = Fxp(x.get_val() - self.get_val(), signed=self.signed or x.signed, n_int=n_int, n_frac=n_frac)
        return y

    __isub__ = __sub__

    def __mul__(self, x):
        if isinstance(x, (int, float)):
            x = Fxp(x)
        
        n_word = self.n_word + x.n_word
        n_frac = self.n_frac + x.n_frac

        y = Fxp(self.get_val() * x.get_val(), signed=self.signed or x.signed, n_word=n_word, n_frac=n_frac)
        return y

    __rmul__ = __mul__

    __imul__ = __mul__

    def __truediv__(self, x):
        if isinstance(x, (int, float)):
            x = Fxp(x)

        y = Fxp(self.get_val() / x.get_val(), signed=self.signed or x.signed)
        return y

    def __rtruediv__(self, x):
        if isinstance(x, (int, float)):
            x = Fxp(x)

        y = Fxp(x.get_val() / self.get_val(), signed=self.signed or x.signed)
        return y

    __itruediv__ = __truediv__

    def __floordiv__(self, x):
        if isinstance(x, (int, float)):
            x = Fxp(x)

        y = Fxp(self.get_val() // x.get_val(), signed=self.signed or x.signed)
        return y

    def __rfloordiv__(self, x):
        if isinstance(x, (int, float)):
            x = Fxp(x)

        y = Fxp(x.get_val() // self.get_val(), signed=self.signed or x.signed)
        return y

    __ifloordiv__ = __floordiv__

    def __mod__(self, x):
        if isinstance(x, (int, float)):
            x = Fxp(x)
        
        n_frac = max(self.n_frac, x.n_frac)
        if self.signed or x.signed:
            n_int = max(self.n_int, x.n_int)  # because python modulo implementation
        else:
            n_int = min(self.n_int, x.n_int)

        y = Fxp(self.get_val() % x.get_val(), signed=self.signed or x.signed, n_word=None, n_frac=n_frac, n_int=n_int)
        return y

    def __rmod__(self, x):
        if isinstance(x, (int, float)):
            x = Fxp(x)
        
        n_frac = max(self.n_frac, x.n_frac)
        if self.signed or x.signed:
            n_int = max(self.n_int, x.n_int)  # because python modulo implementation
        else:
            n_int = min(self.n_int, x.n_int)

        y = Fxp(x.get_val() % self.get_val(), signed=self.signed or x.signed, n_word=None, n_frac=n_frac, n_int=n_int)
        return y

    __imod__ = __mod__

    def __pow__(self, n):
        n_word = self.n_word * n
        n_frac = self.n_frac * n

        y = Fxp(self.get_val() ** n, signed=self.signed or n.signed, n_word=n_word, n_frac=n_frac)
        return y

    def __rpow__(self, n):
        n_word = self.n_word * n
        n_frac = self.n_frac * n

        y = Fxp(n ** self.get_val(), signed=self.signed or n.signed, n_word=n_word, n_frac=n_frac)
        return y

    __ipow__ = __pow__

    
    # bit level operators

    def __rshift__(self, n):
        y = self.deepcopy()
        y.val = y.val >> np.array(n, dtype=y.val.dtype)
        return y

    __irshift__ = __rshift__

    def __lshift__(self, n):
        if self.shifting == 'expand':
            n_word = max(self.n_word, int(np.max(np.ceil(np.log2(np.abs(self.val)+0.5)))) + self.signed + n)
        else:
            n_word = self.n_word

        y = Fxp(None, signed=self.signed, n_word=n_word, n_frac=self.n_frac)
        y.set_val(self.val << np.array(n, dtype=self.val.dtype), raw=True, vdtype=self.vdtype)   # set raw val shifted
        return y
    
    __ilshift__ = __lshift__

    def __invert__(self):
        # inverted_val = ~ self.val

        inverted_val = utils.binary_invert(self.val, n_word=self.n_word)
        if self.signed:
            inverted_val = utils.twos_complement_repr(inverted_val, nbits=self.n_word)
        
        y = self.deepcopy()
        y.set_val(inverted_val, raw=True, vdtype=self.vdtype)   # set raw val inverted
        return y

    def __and__(self, x):
        if isinstance(x, Fxp):
            if self.n_word != x.n_word:
                raise ValueError("Operands dont't have same word size!")
            else:
                x_val = x.val.astype(self.val.dtype) # if it doen't care data type difference
        else:
            x_val = x

        added_val = utils.binary_and(self.val, x_val, n_word=self.n_word)
        if self.signed:
            added_val = utils.twos_complement_repr(added_val, nbits=self.n_word)

        y = self.deepcopy()
        y.set_val(added_val, raw=True, vdtype=self.vdtype)   # set raw val with AND operation  
        return y

    __rand__ = __and__

    __iand__ = __and__

    def __or__(self, x):
        if isinstance(x, Fxp):
            if self.n_word != x.n_word:
                raise ValueError("Operands dont't have same word size!")
            else:
                x_val = x.val.astype(self.val.dtype) # if it doen't care data type difference
        else:
            x_val = x

        ored_val = utils.binary_or(self.val.astype(self.val.dtype), x_val, n_word=self.n_word)
        if self.signed:
            ored_val = utils.twos_complement_repr(ored_val, nbits=self.n_word)

        y = self.deepcopy()
        y.set_val(ored_val, raw=True, vdtype=self.vdtype)   # set raw val with OR operation  
        return y

    __ror__ = __or__

    __ior__ = __or__

    def __xor__(self, x):
        if isinstance(x, Fxp):
            if self.n_word != x.n_word:
                raise ValueError("Operands dont't have same word size!")
            else:
                x_val = x.val.astype(self.val.dtype) # if it doen't care data type difference
        else:
            x_val = x

        xored_val = utils.binary_xor(self.val.astype(self.val.dtype), x_val, n_word=self.n_word)
        if self.signed:
            xored_val = utils.twos_complement_repr(xored_val, nbits=self.n_word)

        y = self.deepcopy()
        y.set_val(xored_val, raw=True, vdtype=self.vdtype)   # set raw val with XOR operation  
        return y  

    __rxor__ = __xor__

    __ixor__ = __xor__    


    # comparisons

    def __lt__(self, x):
        if isinstance(x, Fxp):
            x = x.get_val()
        return self.get_val() < x

    def __le__(self, x):
        if isinstance(x, Fxp):
            x = x.get_val()
        return self.get_val() <= x

    def __eq__(self, x):
        if isinstance(x, Fxp):
            x = x.get_val()
        return self.get_val() == x

    def __ne__(self, x):
        if isinstance(x, Fxp):
            x = x.get_val()
        return self.get_val() != x

    def __gt__(self, x):
        if isinstance(x, Fxp):
            x = x.get_val()
        return self.get_val() > x

    def __ge__(self, x):
        if isinstance(x, Fxp):
            x = x.get_val()
        return self.get_val() >= x

    # indexation
    def __getitem__(self, index):
        return self.get_val()[index]

    def __setitem__(self, index, value):
        new_vals = self.astype(type(value))
        new_vals[index] = value
        self.set_val(new_vals)

    # get info about me
    def get_status(self, format=dict):
        s = None
        if format == dict:
            s = self.status
        elif format == str:
            s = ''
            for k, v in self.status.items():
                if v:
                    s += '\t{:<8}\t=\t{}\n'.format(k,v)
        return s

    def info(self, verbose=1):
        s = ''
        if verbose > 0:
            s += '\tdtype\t\t=\t{}\n'.format(self.dtype)
            s += '\tValue\t\t=\t' + self.__str__() + '\n'
            if self.scaled:
                s += '\tScaling\t\t=\t{} * val + {}\n'.format(self.scale, self.bias)
            s += self.get_status(format=str)
        if verbose > 1:
            s += '\n\tSigned\t\t=\t{}\n'.format(self.signed)
            s += '\tWord bits\t=\t{}\n'.format(self.n_word)
            s += '\tFract bits\t=\t{}\n'.format(self.n_frac)
            s += '\tInt bits\t=\t{}\n'.format(self.n_int)
            s += '\tVal data type\t=\t{}\n'.format(self.vdtype)
        if verbose > 2:
            s += '\n\tUpper\t\t=\t{}\n'.format(self.upper)
            s += '\tLower\t\t=\t{}\n'.format(self.lower)
            s += '\tPrecision\t=\t{}\n'.format(self.precision)
            s += '\tOverflow\t=\t{}\n'.format(self.overflow)
            s += '\tRounding\t=\t{}\n'.format(self.rounding)
            s += '\tShifting\t=\t{}\n'.format(self.shifting)
        print(s)


    # base representations
    def bin(self, frac_dot=False):
        if frac_dot:
            n_frac_dot = self.n_frac
        else:
            n_frac_dot = None
        
        if isinstance(self.val, (list, np.ndarray)) and self.val.ndim > 0:
            if self.vdtype == complex:
                rval = [ utils.binary_repr(int(val.real), n_word=self.n_word, n_frac=n_frac_dot) + '+' + utils.binary_repr(int(val.imag), n_word=self.n_word, n_frac=n_frac_dot) + 'j' for val in self.val]
            else:
                rval = [utils.binary_repr(val, n_word=self.n_word, n_frac=n_frac_dot) for val in self.val]
        else:
            if self.vdtype == complex:
                rval = utils.binary_repr(int(self.val.real), n_word=self.n_word, n_frac=n_frac_dot) + '+' + utils.binary_repr(int(self.val.imag), n_word=self.n_word, n_frac=n_frac_dot) + 'j'
            else:
                rval = utils.binary_repr(self.val, n_word=self.n_word, n_frac=n_frac_dot)
        return rval

    def hex(self):
        if isinstance(self.val, (list, np.ndarray)) and self.val.ndim > 0:
            if self.vdtype == complex:
                rval = [ hex(int(val.split('+')[0], 2)) + '+' +  hex(int(val.split('+')[1][:-1], 2)) + 'j' for val in self.bin()]
            else:
                rval = [hex(int(val, 2)) for val in self.bin()]
        else:
            if self.vdtype == complex:
                rval = hex(int(self.bin().split('+')[0], 2)) + '+' +  hex(int(self.bin().split('+')[1][:-1], 2)) + 'j'
            else:
                rval = hex(int(self.bin(), 2))
        return rval
    
    def base_repr(self, base, frac_dot=False):
        if frac_dot:
            n_frac_dot = self.n_frac
        else:
            n_frac_dot = None

        if isinstance(self.val, (list, np.ndarray)) and self.val.ndim > 0:
            if self.vdtype == complex:
                rval = [utils.base_repr(int(val.real), base=base, n_frac=n_frac_dot) + ('+' if val.imag >= 0 else '') + utils.base_repr(int(val.imag), base=base, n_frac=n_frac_dot) + 'j' for val in self.val]
            else:
                rval = [utils.base_repr(val, base=base, n_frac=n_frac_dot) for val in self.val]
        else:
            if self.vdtype == complex:
                rval = utils.base_repr(int(self.val.real), base=base, n_frac=n_frac_dot) + ('+' if self.val.imag >= 0 else '') + utils.base_repr(int(self.val.imag), base=base, n_frac=n_frac_dot) + 'j'
            else:
                rval = utils.base_repr(self.val, base=base, n_frac=n_frac_dot)
        return rval

    # copy
    def copy(self):
        return copy.copy(self)

    def deepcopy(self):
        return copy.deepcopy(self)

    def like(self, x):
        return  x.copy().set_val(self.get_val()) 

    # reset
    def reset(self):
        #status (overwrite)
        self.status = {
            'overflow': False,
            'underflow': False,
            'inaccuracy': False}        

