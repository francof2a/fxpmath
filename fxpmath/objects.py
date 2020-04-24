import numpy as np 
import copy
from .utils import twos_complement

class Fxp():
    def __init__(self, val=None, signed=None, n_word=None, n_frac=None, n_int=None, 
                max_error=1.0e-6, n_word_max=64):
        self.dtype = 'fxp' # fxp-<sign><n_word>/<n_frac>-{complex}. i.e.: fxp-s16/15, fxp-u8/1, fxp-s32/24-complex
        # value
        self.vdtype = None # value(s) dtype to return as default
        self.val = None
        self.real = None
        self.imag = None
        # format
        self.signed = signed
        self.n_word = n_word
        self.n_frac = n_frac
        # format properties
        self.upper = None
        self.lower = None
        self.precision = None
        #status
        self.status = {
            'overflow': False,
            'underflow': False}
        # behavior
        self.overflow = 'saturate'
        self.rounding = 'trunc'
        # size
        self._init_size(val, signed, n_word, n_frac, n_int, max_error=max_error, n_word_max=n_word_max) 
        # store the value
        self.set_val(val)

    # methods about size
    def _init_size(self, val=None, signed=None, n_word=None, n_frac=None, n_int=None, max_error=1.0e-6, n_word_max=64):
        # sign by default
        if self.signed is None:
            self.signed = True
        
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

            if isinstance(val, (list, np.ndarray)):
                int_vals = np.abs(val).astype(int)
                max_int_val = np.max(int_vals)
                frac_vals = np.subtract(np.abs(val), int_vals)
            elif isinstance(val, (int, float)):
                max_int_val = max([abs(int(val)), 1])
                frac_vals = [np.abs(val - int(val))]
            else:
                raise TypeError('Type not supported for val parameter!')

            if n_word is None and n_frac is None:
                n_int = np.ceil(np.log2(max_int_val)).astype(int) 
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
                n_int = np.ceil(np.log2(max_int_val)).astype(int) + 1
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

        val = np.array(val)

        if self.signed:
            val_max = (1 << (self.n_word-1)) - 1
            val_min = -val_max - 1
        else:
            val_max =  (1 << self.n_word) - 1
            val_min = 0

        if val.dtype != complex:
            new_val = self._round(val * 2.0**self.n_frac, method=self.rounding).astype(int)
            self.val = self._overflow_action(new_val, val_min, val_max)
            self.real = None
            self.imag = None
        else:
            new_val_real = self._round(val.real * 2.0**self.n_frac, method=self.rounding).astype(int)
            new_val_imag = self._round(val.imag * 2.0**self.n_frac, method=self.rounding).astype(int)
            new_val_real = self._overflow_action(new_val_real, val_min, val_max)
            new_val_imag = self._overflow_action(new_val_imag, val_min, val_max)
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
        elif dtype == int:
            val = (self.val // 2.0**self.n_frac).astype(int)
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
        if np.any(new_val > val_max):
            self.status['overflow'] = True
        if np.any(new_val < val_min):
            self.status['underflow'] = True
        
        if self.overflow == 'saturate':
            val = np.clip(new_val, val_min, val_max).astype(int)
        elif self.overflow == 'wrap':
            if new_val.ndim == 0:
                if not ((new_val <= val_max) & (new_val >= val_min)):
                    val = twos_complement(new_val, self.n_word)
            else:
                val = np.array([v if ((v <= val_max) & (v >= val_min)) else twos_complement(v, self.n_word) for v in new_val])
        return val

    def _round(self, val, method='floor'):
        if method == 'around':
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
        n_frac = max(self.n_frac, x.n_frac)

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
    def bin(self):
        if isinstance(self.val, (list, np.ndarray)):
            if self.vdtype == complex:
                rval = [ np.binary_repr(int(val.real), width=self.n_word) + '+' + np.binary_repr(int(val.imag), width=self.n_word) + 'j' for val in self.val]
            else:
                rval = [np.binary_repr(val, width=self.n_word) for val in self.val]
        else:
            if self.vdtype == complex:
                rval = np.binary_repr(int(self.val.real), width=self.n_word) + '+' + np.binary_repr(int(self.val.imag), width=self.n_word) + 'j'
            else:
                rval = np.binary_repr(self.val, width=self.n_word)
        return rval

    def hex(self):
        if isinstance(self.val, (list, np.ndarray)):
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
    
    def base_repr(self, base):
        if isinstance(self.val, (list, np.ndarray)):
            if self.vdtype == complex:
                rval = [np.base_repr(int(val.real), base=base) + ('+' if val.imag >= 0 else '') + np.base_repr(int(val.imag), base=base) + 'j' for val in self.val]
            else:
                rval = [np.base_repr(val, base=base) for val in self.val]
        else:
            if self.vdtype == complex:
                rval = np.base_repr(int(self.val.real), base=base) + ('+' if self.val.imag >= 0 else '') + np.base_repr(int(self.val.imag), base=base) + 'j'
            else:
                rval = np.base_repr(self.val, base=base)
        return rval

    # copy
    def copy(self):
        return copy.copy(self)

    def deepcopy(self):
        return copy.deepcopy(self)

    def like(self, x):
        return  x.copy().set_val(self.get_val()) 


