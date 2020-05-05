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
        # format
        self.signed = None
        self.n_word = None
        self.n_frac = None
        # format properties
        self.upper = None
        self.lower = None
        self.precision = None
        #status
        self.status = None
        # behavior
        self.overflow = None
        self.rounding = None
        # size
        max_error = None
        n_word_max = None

        # check if init must be a `like` other Fxp
        init_like = kwargs.pop('like', None) 
        if init_like is not None:
            if isinstance(init_like, Fxp):
                self.__dict__ = copy.deepcopy(init_like.__dict__)

        #status (overwrite)
        self.status = {
            'overflow': False,
            'underflow': False}
        # behavior
        if self.overflow is None: self.overflow = kwargs.pop('overflow', 'saturate')
        if self.rounding is None: self.rounding = kwargs.pop('rounding', 'trunc')
        # size
        if init_like is None:
            if max_error is None: max_error = kwargs.pop('max_error', 1.0e-6)
            if n_word_max is None: n_word_max = kwargs.pop('n_word_max', 64)
            self._init_size(val, signed, n_word, n_frac, n_int, max_error=max_error, n_word_max=n_word_max)

        # store the value
        self.set_val(val)


    # methods about size
    def _init_size(self, val=None, signed=None, n_word=None, n_frac=None, n_int=None, max_error=1.0e-6, n_word_max=64):
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
        if n_word is None or n_frac is None or val is None:
            self.set_best_sizes(val, n_word, n_frac, max_error=max_error, n_word_max=n_word_max)
        else:
            self.resize(self.signed, n_word, n_frac, n_int)


    def resize(self, signed=None, n_word=None, n_frac=None, n_int=None):
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

        # re store the value
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

            # if val is a str(s), convert to number(s)
            val = utils.str2num(val, self.signed, self.n_word, self.n_frac)

            if isinstance(val, (list, np.ndarray)):
                int_vals = val.astype(int)
                max_int_val = np.max(np.abs(int_vals + 0.5))
                frac_vals = np.abs(np.subtract(val, int_vals))
            elif isinstance(val, (int, float)):
                max_int_val = abs(val + 0.5)
                frac_vals = [np.abs(val - int(val))]
            else:
                raise TypeError('Type not supported for val parameter!')

            if n_word is None and n_frac is None:
                n_int = max( np.ceil(np.log2(max_int_val)).astype(int), 0) 
                max_n_frac = n_word_max - n_int - sign

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
                self.n_frac = max(n_frac_calcs)
                self.n_word = int(n_frac + n_int + sign)
            elif n_word is None:
                n_int = max( np.ceil(np.log2(max_int_val)).astype(int), 0)
                self.n_frac = n_frac
                self.n_word = int(n_frac + n_int + sign)
            elif n_frac is None:
                self.n_word = n_word
                n_int = np.ceil(np.log2(np.abs(val)))
                self.n_frac = np.max(n_word-n_int-sign, 0).astype(int)
        
        self.n_word = min(self.n_word, n_word_max)
        self.resize()

    # methods about value

    def set_val(self, val):
        if val is None:
            val = 0

        # if val is a str(s), convert to number(s)
        val = utils.str2num(val, self.signed, self.n_word, self.n_frac)
        # convert to (numpy) ndarray
        val = np.array(val)

        if self.signed:
            val_max = (1 << (self.n_word-1)) - 1
            val_min = -val_max - 1
            val_dtype = 'int64'
        else:
            val_max =  (1 << self.n_word) - 1
            val_min = 0
            val_dtype = 'uint64'

        if val.dtype != complex:
            new_val = self._round(val * 2**self.n_frac , method=self.rounding)
            new_val = self._overflow_action(new_val, val_min, val_max)
            self.val = new_val.astype(val_dtype)
            self.real = None
            self.imag = None
        else:
            new_val_real = self._round(val.real * 2**self.n_frac, method=self.rounding)
            new_val_imag = self._round(val.imag * 2**self.n_frac, method=self.rounding)
            new_val_real = self._overflow_action(new_val_real, val_min, val_max).astype(val_dtype)
            new_val_imag = self._overflow_action(new_val_imag, val_min, val_max).astype(val_dtype)
            self.val = new_val_real + 1j * new_val_imag
            self.real = self.astype(complex).real
            self.imag = self.astype(complex).imag

        # dtype
        self.dtype = 'fxp-{sign}{nword}/{nfrac}{comp}'.format(sign='s' if self.signed else 'u', 
                                                             nword=self.n_word, 
                                                             nfrac=self.n_frac, 
                                                             comp='-complex' if val.dtype == complex else '')

        # dtype_return (default)
        self.vdtype = val.dtype

        return self

    def astype(self, dtype=None):
        if dtype is None:
            dtype = self.vdtype

        if dtype == float:
            val = self.val / 2.0**self.n_frac
        elif dtype == int or dtype == 'uint':
            val = self.val.astype(dtype) // 2**self.n_frac
        elif dtype == complex:
            val = (self.val.real + 1j * self.val.imag) / 2.0**self.n_frac
        else:
            val = None
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
        if np.any(new_val.any() > val_max):
            self.status['overflow'] = True
        if np.any(new_val.any() < val_min):
            self.status['underflow'] = True
        
        if self.overflow == 'saturate':
            #val = np.clip(new_val, val_min, val_max) # it returns float that cause an error for 64 bits huge integers
            if new_val.ndim > 0:
                val = np.array([max(val_min, min(val_max, v)) for v in new_val])
            else:
                val = np.array(max(val_min, min(val_max, new_val)))
        elif self.overflow == 'wrap':
            if new_val.ndim == 0:
                if not ((new_val <= val_max) & (new_val >= val_min)):
                    val = utils.twos_complement(new_val, self.n_word)
                else:
                    val = new_val
            else:
                val = np.array([v if ((v <= val_max) & (v >= val_min)) else utils.twos_complement(v, self.n_word) for v in new_val])
        return val

    def _round(self, val, method='floor'):
        if val.dtype == int  or val.dtype == 'uint':
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
        
        n_word = max(self.n_word, x.n_word) + 1
        n_frac = max(self.n_frac, x.n_frac)

        y = Fxp(self.astype(float) + x.astype(float), signed=self.signed or x.signed, n_word=n_word, n_frac=n_frac)
        return y

    def __sub__(self, x):
        if isinstance(x, (int, float, list, np.ndarray)):
            x = Fxp(x)
        
        n_word = max(self.n_word, x.n_word) + 1
        n_frac = max(self.n_frac, x.n_frac)

        y = Fxp(self.astype(float) - x.astype(float), signed=self.signed or x.signed, n_word=n_word, n_frac=n_frac)
        return y

    def __mul__(self, x):
        if isinstance(x, (int, float)):
            x = Fxp(x)
        
        n_word = self.n_word + x.n_word
        n_frac = self.n_frac + x.n_frac

        y = Fxp(self.astype(float) * x.astype(float), signed=self.signed or x.signed, n_word=n_word, n_frac=n_frac)
        return y

    def __rmul__(self, x):
        return self * x

    def __truediv__(self, x):
        if isinstance(x, (int, float)):
            x = Fxp(x)

        y = Fxp(self.astype(float) / x.astype(float), signed=self.signed or x.signed)
        return y

    def __floordiv__(self, x):
        if isinstance(x, (int, float)):
            x = Fxp(x)

        y = Fxp(self.astype(float) // x.astype(float), signed=self.signed or x.signed)
        return y

    def __mod__(self, x):
        if isinstance(x, (int, float)):
            x = Fxp(x)
        
        n_frac = max(self.n_frac, x.n_frac)
        n_int = min(self.n_int, x.n_int)

        y = Fxp(self.astype(float) % x.astype(float), signed=self.signed or x.signed, n_word=None, n_frac=n_frac, n_int=n_int)
        return y        

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

    def info(self):
        s = ''
        s += '\tdtype\t\t=\t{}\n'.format(self.dtype)
        s += '\tValue\t\t=\t' + self.__str__() + '\n'
        s += '\tSigned\t\t=\t{}\n'.format(self.signed)
        s += '\tWord bits\t=\t{}\n'.format(self.n_word)
        s += '\tFract bits\t=\t{}\n'.format(self.n_frac)
        s += '\tInt bits\t=\t{}\n'.format(self.n_int)
        s += self.get_status(format=str)
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


