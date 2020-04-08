import numpy as np 
from .utils import twos_complement

class Fxp():
    def __init__(self, val=None, signed=None, n_word=None, n_frac=None, 
                max_error=1.0e-6, n_word_max=64):
        self.dtype = 'fxp' # fxp-<sign><n_word>/<n_frac>-{complex}. i.e.: fxp-s16/15, fxp-u8/1, fxp-s32/24-complex
        self.val = None
        self.real = None
        self.imag = None
        self.signed = signed
        self.n_word = n_word
        self.n_frac = n_frac
        self.status = {
            'overflow': False,
            'underflow': False}

        self.props = {
            'overflow': 'saturate',
            'rounding': 'trunc'}

        if self.signed is None:
            self.signed = True
        
        if n_word is None or n_frac is None or val is None:
            self.set_best_sizes(val, n_word, n_frac, 
                                max_error=max_error, n_word_max=n_word_max)
        else:
            self.n_word = n_word
            self.n_frac = n_frac

        self.set_val(val)
    
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
                max_int_val = abs(int(val))
                frac_vals = [np.abs(val - int(val))]
            else:
                raise TypeError('Type not supported for val parameter!')

            if n_word is None and n_frac is None:
                n_int = np.ceil(np.log2(max_int_val)).astype(int) + 1 
                max_n_frac = n_word_max - n_int - sign

                n_frac_calcs = []
                for r in frac_vals:
                    e = 1.0
                    n_frac = 0
                    while e > max_error and n_frac <= max_n_frac:
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
            new_val = self._round(val * 2.0**self.n_frac, method=self.props['rounding']).astype(int)
            self.val = self._overflow_action(new_val, val_min, val_max)
            self.real = None
            self.imag = None
        else:
            new_val_real = self._round(val.real * 2.0**self.n_frac, method=self.props['rounding']).astype(int)
            new_val_imag = self._round(val.imag * 2.0**self.n_frac, method=self.props['rounding']).astype(int)
            new_val_real = self._overflow_action(new_val_real, val_min, val_max)
            new_val_imag = self._overflow_action(new_val_imag, val_min, val_max)
            self.val = new_val_real + 1j * new_val_imag
            self.real = self.astype(complex).real
            self.imag = self.astype(complex).imag

        self.dtype = 'fxp-{sign}{nword}/{nfrac}{comp}'.format(sign='s' if self.signed else 'u', 
                                                             nword=self.n_word, 
                                                             nfrac=self.n_frac, 
                                                             comp='-complex' if val.dtype == complex else '')
        return self.val

    def _overflow_action(self, new_val, val_min, val_max):
        if np.any(new_val > val_max):
            self.status['overflow'] = True
        if np.any(new_val < val_min):
            self.status['underflow'] = True
        
        if self.props['overflow'] == 'saturate':
            val = np.clip(new_val, val_min, val_max).astype(int)
        elif self.props['overflow'] == 'wrap':
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
    
    def astype(self, dtype):
        if dtype == float:
            val = self.val / 2.0**self.n_frac
        elif dtype == int:
            val = int(self.val // 2.0**self.n_frac)
        elif dtype == complex:
            val = (self.val.real + 1j * self.val.imag) / 2.0**self.n_frac
        return val

    def __repr__(self):
        if self.dtype.endswith('complex'):
            s = str(self.astype(complex))
        else:
            s = str(self.astype(float))
        return s

    def __str__(self):
        if self.dtype.endswith('complex'):
            s = str(self.astype(complex))
        else:
            s = str(self.astype(float))
        return s

    def __add__(self, x):
        if isinstance(x, (int, float, list, np.ndarray)):
            x = Fxp(x, signed=self.signed, n_word=self.n_word, n_frac=self.n_frac)
        
        n_word = max(self.n_word, x.n_word) + 1
        n_frac = max(self.n_frac, x.n_frac)

        y = Fxp(self.astype(float) + x.astype(float), signed=self.signed or x.signed, n_word=n_word, n_frac=n_frac)
        return y

    def __sub__(self, x):
        if isinstance(x, (int, float, list, np.ndarray)):
            x = Fxp(x, signed=self.signed, n_word=self.n_word, n_frac=self.n_frac)
        
        n_word = max(self.n_word, x.n_word) + 1
        n_frac = max(self.n_frac, x.n_frac)

        y = Fxp(self.astype(float) - x.astype(float), signed=self.signed or x.signed, n_word=n_word, n_frac=n_frac)
        return y

    def __mul__(self, x):
        if isinstance(x, (int, float)):
            x = Fxp(x, signed=self.signed, n_word=self.n_word, n_frac=self.n_frac)
        
        n_word = self.n_word + x.n_word
        n_frac = max(self.n_frac, x.n_frac)

        y = Fxp(self.astype(float) * x.astype(float), signed=self.signed or x.signed, n_word=n_word, n_frac=n_frac)
        return y

    def __rmul__(self, x):
        return self * x

    def equal(self, x):
        if isinstance(x, Fxp):
            x = x.astype(float)
        self.set_val(x)
        return self

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
        s += self.get_status(format=str)
        return s


