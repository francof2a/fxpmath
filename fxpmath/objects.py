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
import math
import copy
import re

from . import utils
from . import _n_word_max, _max_error

_NUMPY_HANDLED_FUNCTIONS = {}

try:
    from decimal import Decimal
    from decimal import getcontext
except:
    Decimal = type(None)

#%%
class Fxp():
    '''
    Numerical Fractional Fixed-Point object (base 2).

    Parameters
    ---

    val : None, int, float, complex, list of numbers, numpy array, str (bin, hex, dec), optional, default=None
        Value(s) to be stored in fractional fixed-point (base 2) format.
    
    signed : bool, optional, default=None
        If True, a sign bit is used for the binary word. If None, Fxp is signed.

    n_word : int, optional, defualt=None
        Number of the bits for binary word (sign + integer part + fractional part).
        If None, best word size is calculated according input value(s) and other sizes defined.

    n_frac : int, optional, default=None
        Number of bits for fractional part.
        If None, best word size is calculated according input value(s) and other sizes defined.

    n_int : int, optional, default=None
        Number of bits for integer part.
        If None, best word size is calculated according input value(s) and other sizes defined.

    like : Fxp, optional, default=None
        Init new Fxp object using all parameters of `like` Fxp object, except its value.
        
    dtype : str, optional, default=None
        String describing the desired fixed-point format in either Q/UQ, S/U, or fxp dtype format.

    **kwargs : alternative keywords parameters.

    Attributes
    ---

    dtype : str, read only.
        String describing the fixed-point format in either Q/UQ, S/U, or fxp dtype format (default).
        Set config.dtype_notation to change format.

    vdtype : type.
        Data type of the original value.

    val : number or array
        Value represented in original format (not binary).

    real : number or array
        Value represented in original format (not binary).
        Real part when Fxp is a complex value.

    imag : number or array
        Imaginary part when Fxp is a complex value.
        Equal to zero when Fxp is a real value.

    upper : number
        Maximum value that can be represented by this Fxp.

    lower : number
        Minimum value that can be represented by this Fxp.

    precision : number
        Resolution of the values that can be represented by this Fxp.

    shape : tuple
        Shape of the array of values. Empty if Fxp is a single value.

    ndim : int
        Number of dimensions of the array of values. It's equal to 0 if Fxp is a single value.

    size : int
        Number of values in this Fxp object.

    config : Config class
        Class where configurations parameters of this Fxp object are stored.

    callbacks : list
        List of callbacks.

    '''

    template = None

    def __init__(self, val=None, signed=None, n_word=None, n_frac=None, n_int=None, like=None, dtype=None, **kwargs):

        # Init all properties in None
        self._dtype = 'fxp' # fxp-<sign><n_word>/<n_frac>-{complex}. i.e.: fxp-s16/15, fxp-u8/1, fxp-s32/24-complex
        # value
        self.vdtype = None # value(s) dtype to return as default
        self.val = None
        self.real = None
        self.imag = None
        raw = None
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
        self.callbacks = None
        #config
        self.config = None

        _initialized = False
        # ---

        # init config
        self.config = Config()

        # if `template` is in kwarg, the reference template is updated
        if 'template' in kwargs: self.template = kwargs.pop('template')

        # check if init must be a `like` other Fxp
        if like is not None:
            if isinstance(like, Fxp):
                self.__dict__ = copy.deepcopy(like.__dict__)
                self.val = None
                self.real = None
                self.imag = None
                _initialized = True

        elif self.template is not None:
            # init must be a `like` template Fxp
            if isinstance(self.template, Fxp):
                self.__dict__ = copy.deepcopy(self.template.__dict__)
                self.val = None
                self.real = None
                self.imag = None
                _initialized = True

        #status (overwrite)
        self.status = {
            'overflow': False,
            'underflow': False,
            'inaccuracy': False,
            'extended_prec': False}

        # update config as argument
        _config = kwargs.pop('config', None)
        if _config is not None:
            if isinstance(_config, Config):
                self.config = _config.deepcopy()
            else:
                raise TypeError('config parameter must be Config class type!')

        # update config from kwargs
        self.config.update(**kwargs)

        # callbacks
        if self.callbacks is None: self.callbacks = kwargs.pop('callbacks', [])

        # scaling
        if self.scale is None: self.scale = kwargs.pop('scale', 1)
        if self.bias is None: self.bias = kwargs.pop('bias', 0)
        self.scaled = True if self.scale != 1 or self.bias != 0 else False

        # check if val is a raw value
        if raw is None: raw = kwargs.pop('raw', False)

        # check if a string-based format has been provided
        if dtype is not None:
            signed, n_word, n_frac, complex_flag = self._parseformatstr(dtype)

            self.vdtype = complex if complex_flag else self.vdtype

        # size
        if not _initialized:
            self._init_size(val, signed, n_word, n_frac, n_int, max_error=self.config.max_error, n_word_max=self.config.n_word_max, raw=raw)
        else:
            # overwrite with other sizes if some are not None
            self.resize(signed, n_word, n_frac, n_int)

        # update dtype
        self._update_dtype()

        # store the value
        self.set_val(val, raw=raw)

    # ---
    # Properties/Attributes
    # ---
    # region

    @property
    def dtype(self):
        return self._dtype

    # overflow (mirror of config for compatibility)
    @property
    def overflow(self):
        return self.config.overflow
    
    @overflow.setter
    def overflow(self, val):
        self.config.overflow = val

    # rounding (mirror of config for compatibility)
    @property
    def rounding(self):
        return self.config.rounding
    
    @rounding.setter
    def rounding(self, val):
        self.config.rounding = val

    # shifting (mirror of config for compatibility)
    @property
    def shifting(self):
        return self.config.shifting
    
    @shifting.setter
    def shifting(self, val):
        self.config.shifting = val

    @property
    def shape(self):
        return self.val.shape

    @property
    def ndim(self):
        return self.val.ndim

    @property
    def size(self):
        return self.val.size


    # endregion

    # ---
    # Methods
    # ---
    # region

    # methods about size

    def get_dtype(self, notation=None):
        """
        Get dtype attribute of Fxp.

        Parameters
        ---

        notation : {None, 'fxp', 'Q'} str, default=None
            Set format of notation used for dtype.
            If `notation=None`, value in config.dtype_notation is used.
            If `notation='Q'`, dtype format will be Q Format; else 'fxp' is used by default.

        Returns
        ---
        dtype : str
            String describing the fixed-point format in either Q/UQ, S/U, or fxp dtype format (default).

        """
        self._update_dtype(notation)    # update dtype
        return self._dtype
    
    def _qfmt(self):
        return re.compile(r'(s|u|q|uq|qu)(\d+)(\.[+-]?\d+)?')
    
    def _fxpfmt(self):
        return re.compile(r'fxp-(s|u)(\d+)/([+-]?\d+)(-complex)?')
    
    def _parseformatstr(self, fmt):
        fmt = fmt.casefold()
        mo = self._qfmt().match(fmt)
        if mo:
            # Q/S notation counts the sign bit as an integer bit, such that
            # the total number of bits is always int+frac
            signed = mo.group(1) in 'sq'
            n_int = int(mo.group(2))
            if mo.group(3) is None:
                n_frac = 0
            else:
                n_frac = int(mo.group(3)[1:])
            n_word = n_frac + n_int
            complex_dtype = False
        else:
            mo = self._fxpfmt().match(fmt)
            if mo:
                signed = mo.group(1) == 's'
                n_word = int(mo.group(2))
                n_frac = int(mo.group(3))
                complex_dtype = False
                if mo.lastindex > 3:
                    _complex_str = str(mo.group(4))
                    if _complex_str == '-complex':
                        complex_dtype = True
                        
            else:
                raise ValueError('unrecognized format string')
        return signed, n_word, n_frac, complex_dtype
    
    def _init_size(self, val=None, signed=None, n_word=None, n_frac=None, n_int=None, max_error=_max_error, n_word_max=_n_word_max, raw=False):
        # check signed type
        if not isinstance(signed, (type(None), bool, int)):
            raise TypeError("signed must be boolean (True, False), int (1 or 0) or None!")
        
        # check n_word, n_frac, n_int type
        if not isinstance(n_word, (type(None), int)):
            raise TypeError("n_word must be integer or None!")
        if not isinstance(n_frac, (type(None), int)):
            raise TypeError("n_frac must be integer or None!")
        if not isinstance(n_int, (type(None), int)):
            raise TypeError("n_int must be integer or None!")
        
        # sign by default
        if signed is None:
            self.signed = True
        else:
            self.signed = bool(signed)
            if self.signed != 0 and self.signed != 1:
                raise ValueError("If signed is int, the valid values are 1 (True) and 0 (False)!")
        
        # n_int defined:
        if n_word is None and n_frac is not None and n_int is not None:
            n_word = n_int + n_frac + (1 if self.signed else 0)
        elif n_frac is None and n_word is not None and n_int is not None:
            n_frac = n_word - n_int - (1 if self.signed else 0)

        # check if I must find the best size for val
        if n_word is None or n_frac is None:
            self.set_best_sizes(val, n_word, n_frac, max_error=max_error, n_word_max=n_word_max, raw=raw)
        else:
            self.resize(self.signed, n_word, n_frac, n_int)

    def resize(self, signed=None, n_word=None, n_frac=None, n_int=None, restore_val=True, dtype=None):
        """
        Change size of one or more of size parameters (signed, n_word, n_int and/or n_frac) of the Fxp object.

        Parameters
        ---

        signed : bool, optional, default=None
            If True, a sign bit is used for the binary word. If None, Fxp is signed.

        n_word : int, optional, defualt=None
            Number of the bits for binary word (sign + integer part + fractional part).
            If None, best word size is calculated according input value(s) and other sizes defined.

        n_frac : int, optional, default=None
            Number of bits for fractional part.
            If None, best word size is calculated according input value(s) and other sizes defined.

        n_int : int, optional, default=None
            Number of bits for integer part.
            If None, best word size is calculated according input value(s) and other sizes defined.

        restore_val : bool
            If `True` restores original value (if it's possible) after size changing, if `False` the raw (integer fixed point) value is kept.

        dtype : str, optional, default=None
                String describing the desired fixed-point format in either Q/UQ, S/U, or fxp dtype format.
                If some of the size parameters are not None, a ValueError exception will be raised.
        """
        _old_val = self.val
        _old_n_frac = self.n_frac

        # check signed type
        if not isinstance(signed, (type(None), bool, int)):
            raise TypeError("signed must be boolean (True, False), int (1 or 0) or None!")
        
        # check n_word, n_frac, n_int type
        if not isinstance(n_word, (type(None), int)):
            raise TypeError("n_word must be integer or None!")
        if not isinstance(n_frac, (type(None), int)):
            raise TypeError("n_frac must be integer or None!")
        if not isinstance(n_int, (type(None), int)):
            raise TypeError("n_int must be integer or None!")
        
        # sign by default
        if signed is not None:
            self.signed = bool(signed)
            if self.signed != 0 and self.signed != 1:
                raise ValueError("If signed is int, the valid values are 1 (True) and 0 (False)!")

        # check if a string-based format has been provided
        if dtype is not None:
            if signed is not None or n_word is not None or n_frac is not None or n_int is not None:
                raise ValueError('If dtype is specified, other sizing parameters must be `None`!')
            signed, n_word, n_frac, complex_flag = self._parseformatstr(dtype)

            self.vdtype = complex if complex_flag else self.vdtype

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
            self.n_word = int(n_word)
        # frac
        if n_frac is not None:
            self.n_frac = int(n_frac)
    
        # n_int    
        self.n_int = self.n_word - self.n_frac - (1 if self.signed else 0)

        # status extended precision
        if self.n_word >= _n_word_max:
            self.status['extended_prec'] = True
        else:
            self.status['extended_prec'] = False

        # upper and lower limits
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
        if restore_val and _old_val is not None and self.n_frac is not None:
            if self.scaled:
                self.set_val((_old_val / 2**_old_n_frac) * self.scale + self.bias)
            else:
                self.set_val(_old_val * 2**(self.n_frac - _old_n_frac), raw=True)
        else:
            self.set_val(_old_val, raw=True)

        # update dtype
        self._update_dtype()
    
    def set_best_sizes(self, val=None, n_word=None, n_frac=None, max_error=1.0e-6, n_word_max=64, raw=False):

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
            val, _, raw, signed, n_word, n_frac = self._format_inupt_val(val, return_sizes=True, raw=raw)
            val = np.array([val])

            # check if val is complex, if it is: convert to array of float/int
            if np.iscomplexobj(val) or isinstance(val.item(0), complex):
                val_real = np.vectorize(lambda v: v.real)(val)
                val_imag = np.vectorize(lambda v: v.imag)(val)
                val = np.array([val_real, val_imag])
            
            # if val is raw
            if raw:
                if self.n_frac is not None:
                    val = val / self._get_conv_factor()
                else:
                    raise ValueError('for raw value, `n_frac` must be defined!')

            # define numpy integer type
            if self.signed:
                int_dtype = np.int64
            else:
                int_dtype = np.uint64

            # find fractional parts
            frac_vals = np.abs(val%1).ravel()

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
            # n_int = max( np.ceil(np.log2(np.max(np.abs( val*(1 << n_frac) + 0.5 )))).astype(int_dtype) - n_frac, 0)
            
            val_max = int(np.max(val)*(1 << n_frac))
            val_min = int(np.min(val)*(1 << n_frac))
            n_int = 0
            while n_int < n_word_max - sign:
                msb_max = (val_max >> n_int) + (1 if val_max < 0 else 0)
                msb_min = (val_min >> n_int) + (1 if val_min < 0 else 0)

                if msb_max == msb_min == 0:
                    break
                n_int += 1

            n_int = max(n_int - n_frac, 0)

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

    def reshape(self, shape, order='C'):
        """
        Reshape the fixed-point array.

        Parameters
        ---
        shape : int or tuple of ints
            The new shape should be compatible with the original shape. 
            If an integer, then the result will be a 1-D array of that length. 
            One shape dimension can be -1. In this case, the value is inferred from the length of the array and remaining dimensions.

        order : {'C', 'F', 'A'}, optional
            Read the elements of a using this index order, and place the elements into the reshaped array using this index order. 
            'C' means to read / write the elements using C-like index order, with the last axis index changing fastest, back to the first axis index changing slowest. 
            'F' means to read / write the elements using Fortran-like index order, with the first index changing fastest, and the last index changing slowest. 
            Note that the 'C' and 'F' options take no account of the memory layout of the underlying array, and only refer to the order of indexing. 
            'A' means to read / write the elements in Fortran-like index order if a is Fortran contiguous in memory, C-like order otherwise.

        Returns
        ---
        reshaped_fxp : Fxp object
            Same Fxp object with its reshaped values array.

        """
        
        self.val = self.val.reshape(shape=shape, order=order)
        return self
    
    def flatten(self, order='C'):
        """
        Return a copy of the Fxp with its values array collapsed into one dimension.

        Parameters
        ---

        order{'C', 'F', 'A', 'K'}, optional
            'C' means to flatten in row-major (C-style) order. 
            'F' means to flatten in column-major (Fortran- style) order. 
            'A' means to flatten in column-major order if a is Fortran contiguous in memory, row-major order otherwise. 
            'K' means to flatten a in the order the elements occur in memory. 
            The default is 'C'.

        Returns
        ---
        x : Fxp object
            A copy of the Fxp object, with its values array flattened to one dimension.

        """

        x = self.copy()
        x.val = x.val.flatten(order)
        return x

    # methods about value

    def _format_inupt_val(self, val, return_sizes=False, raw=False, set_inaccuracy=True):
        vdtype = None
        signed = self.signed
        n_word = self.n_word
        n_frac = self.n_frac

        if val is None:
            val = 0

            if self.vdtype is None:
                vdtype = int if n_frac < 1 else float
            else:
                vdtype = self.vdtype

        elif isinstance(val, Fxp):
            # if val is an Fxp object
            vdtype = val.vdtype
            # if some of signed, n_word, n_frac is not defined, they are copied from val
            if self.signed is None: self.signed = val.signed
            if self.n_word is None: self.n_word = val.n_word
            if self.n_frac is None: self.n_frac = val.n_frac

            # check inaccuracy
            if set_inaccuracy and val.status['inaccuracy']:
                self.status['inaccuracy'] = True

            # force return raw value for better precision
            val = val.val * 2**(self.n_frac - val.n_frac)
            raw = True

        elif isinstance(val, (int, float, complex)):
            vdtype = type(val)

        elif isinstance(val, (np.ndarray, np.generic)):
            if isinstance(val, object):
                vdtype = type(val.item(0))
            else:
                vdtype = val.dtype
            
            try:
                if isinstance(val, np.float128):
                    val = np.array(float(val))
            except:
                # by now it is just an extra test, not critical
                pass

            if np.issubdtype(val.dtype, np.str_):
                # if val is a str(s), convert to number(s)
                val = val.tolist()

                if not raw:
                    val, signed, n_word, n_frac = utils.str2num(val, self.signed, self.n_word, self.n_frac, return_sizes=True)
                else:
                    val, signed, n_word, _ = utils.str2num(val, self.signed, self.n_word, None, return_sizes=True)
                    n_frac = self.n_frac

                if n_frac is not None and n_frac == 0:
                    vdtype = int
                else:
                    vdtype = float

        elif isinstance(val, (list, tuple, str)):
            # if val is a str(s), convert to number(s)
            if not raw:
                val, signed, n_word, n_frac = utils.str2num(val, self.signed, self.n_word, self.n_frac, return_sizes=True)
            else:
                val, signed, n_word, _ = utils.str2num(val, self.signed, self.n_word, None, return_sizes=True)
                n_frac = self.n_frac

        elif isinstance(val, Decimal):
            vdtype = float            # assuming float format

            if self.n_frac is None:
                # estimate n_frac from decimal precision
                self.n_frac = n_frac = int(np.ceil(math.log2(10**int(getcontext().prec))))

            # force return raw value for better precision
            val = int(val * 2**(self.n_frac))
            raw = True

        else:
            raise ValueError('Not supported input type: {}'.format(type(val)))

        # convert to (numpy) ndarray
        val = np.array(val)

        if vdtype is None:
            vdtype = val.dtype
        
        # scaling conversion
        self.scaled = False
        if self.scale is not None and self.bias is not None and not raw:
            if self.bias != 0:
                val = val - self.bias
            if self.scale != 1:
                val = val / self.scale

            if self.bias != 0 or self.scale != 1:
                self.scaled = True # update scaled flag

                # update vdtype due scaling tranformation
                if vdtype == int and (isinstance(self.bias, float) or self.scale != 1):
                    vdtype = float
            
            # check if it is a numpy array
            if not isinstance(val, (np.ndarray, np.generic)):
                val = np.array(val)

        if return_sizes:
            return val, vdtype, raw, signed, n_word, n_frac
        else:
            return val, vdtype, raw

    def _get_conv_factor(self, raw=False):
        # precision_cast = (lambda m: np.array(m, dtype=object)) if self.status['extended_prec'] else (lambda m: m)

        if raw:
            conv_factor = 1
        elif self.n_frac >= 0:
            conv_factor = (1<<self.n_frac)
        else:
            conv_factor = 1/(1<<-self.n_frac)

        return conv_factor

    def _update_dtype(self, notation=None):
        if notation is None:
            notation = self.config.dtype_notation
        else:
            notation = 'fxp'

        if self.signed is not None and self.n_word is not None and self.n_frac is not None:
            if notation == 'Q':
                self._dtype = '{Q}{nint}.{nfrac}'.format(Q='Q' if self.signed else 'UQ',
                                                        nint=self.n_word-self.n_frac,
                                                        nfrac=self.n_frac)
            elif self.val is not None:
                self._dtype = 'fxp-{sign}{nword}/{nfrac}{comp}'.format(sign='s' if self.signed else 'u', 
                                                                    nword=self.n_word, 
                                                                    nfrac=self.n_frac, 
                                                                    comp='-complex' if (isinstance(self.val, complex) or \
                                                                        self.val.dtype == complex or \
                                                                        self.vdtype == complex) else '')
            else:
                self._dtype = 'fxp-{sign}{nword}/{nfrac}'.format(sign='s' if self.signed else 'u', 
                                                                    nword=self.n_word, 
                                                                    nfrac=self.n_frac)                
        else:
            self._dtype = 'fxp'

    def set_val(self, val, raw=False, vdtype=None, index=None):
        """
        Set the value/s of the Fxp object.

        Paramters
        ---

        val : None, int, float, complex, list of numbers, numpy array, str (bin, hex, dec), optional, default=None
            Value(s) to be stored in fractional fixed-point (base 2) format.

        raw : bool, optional, default=False
            If `True` the integer value which represent the fixed-point value is overwritten by `val` input.

        vdtype : type, optional, default=None
            Data type to overwrite Fxp vdtype when a raw value is set (`raw=True`).

        index : int, optional, default=None
            Index of the element to be overwritten in list or array of values by `val` input.

        Returns
        ---

        self : Fxp object
            Fxp with it's value modified. 

        """

        # convert input value to valid format
        val, original_vdtype, raw = self._format_inupt_val(val, raw=raw)

        # val limits according word size
        if self.signed:
            val_max = (1 << (self.n_word-1)) - 1
            val_min = -val_max - 1
        else:
            val_max =  (1 << self.n_word) - 1
            val_min = 0

        # conversion factor
        conv_factor = self._get_conv_factor(raw)

        # round, saturate and store
        if original_vdtype != complex and not np.issubdtype(original_vdtype, np.complexfloating):
            # val_dtype determination
            _n_word_max_ = min(_n_word_max, 64)
            if np.max(val) >= 2**_n_word_max_ or np.min(val) < -2**_n_word_max_ or self.n_word >= _n_word_max_:
                val_dtype = object
                val = val.astype(object)
            else:
                val = val.astype(original_vdtype)
                val_dtype = np.int64 if self.signed else np.uint64

            # rounding and overflowing
            new_val = self._round(val * conv_factor , method=self.config.rounding)
            new_val = self._overflow_action(new_val, val_min, val_max)

            # convert to array of val_dtype
            new_val = new_val.astype(val_dtype)

            if val_dtype == object:       
                # convert each element to int
                new_val = np.array(list(map(int, new_val.flatten()))).reshape(new_val.shape).astype(val_dtype)
            
            if index is not None:
                self.val[index] = new_val
            else:
                self.val = new_val

            self.real = self.get_val()
            self.imag = 0

        else:
            # extract real and imaginary parts
            new_val_real = np.vectorize(lambda v: v.real)(val)
            new_val_imag = np.vectorize(lambda v: v.imag)(val)
            
            # val_dtype determination
            _n_word_max_ = min(_n_word_max, 64)
            if np.max(new_val_real) >= 2**_n_word_max_ or np.min(new_val_real) < -2**_n_word_max_ or self.n_word >= _n_word_max_:
                val_dtype = object
            else:
                val = val.astype(original_vdtype)
                val_dtype = np.int64 if self.signed else np.uint64
            
            # rounding and overflowing
            new_val_real = self._round(new_val_real * conv_factor, method=self.config.rounding)
            new_val_imag = self._round(new_val_imag * conv_factor, method=self.config.rounding)
            new_val_real = self._overflow_action(new_val_real, val_min, val_max)
            new_val_imag = self._overflow_action(new_val_imag, val_min, val_max)

            # convert to array of val_dtype
            new_val_real = new_val_real.astype(val_dtype)
            new_val_imag = new_val_imag.astype(val_dtype)

            if val_dtype == object:       
                # convert each element to int
                new_val_real = np.array(list(map(int, new_val_real.flatten()))).reshape(new_val_real.shape)
                new_val_imag = np.array(list(map(int, new_val_imag.flatten()))).reshape(new_val_imag.shape)
            
            # rebuild complex
            new_val = new_val_real + 1j * new_val_imag

            if index is not None:
                self.val[index] = new_val
            else:
                self.val = new_val

            self.real = self.astype(complex).real
            self.imag = self.astype(complex).imag

        # update dtype
        self._update_dtype()

        # vdtype
        if raw:
            if vdtype is not None:
                self.vdtype = vdtype
        else:
            self.vdtype = original_vdtype
            if np.issubdtype(self.vdtype, np.integer) and self.n_frac > 0:
                self.vdtype = float  # change to float type if Fxp has fractional part

        # check inaccuracy
        if not np.equal(val, new_val/conv_factor).all() :
            self.status['inaccuracy'] = True
            self._run_callbacks('on_status_inaccuracy')

        # run changed value callback
        self._run_callbacks('on_value_change')

        return self

    def astype(self, dtype=None, index=None, item=None):
        """
        Returns non-fixed-point value cast to a specified type.

        Paramters
        ---

        dtype : str or dtype, optional, default=None
            Typecode or data-type to which the array is cast.

            `None` returns according to `vdtype` of Fxp, if last one is `None`, `float` is returned.
        
        index : int, optional, default=None
            Index of the element to return from list or array of values cast according `dtype`.

        item : variable number and type, optional, default=None

            None: value is not returned as an item.

            int_type: this argument is interpreted as a flat index into the array, specifying which element to cast and return.

            tuple of int_types: functions as does a single int_type argument, except that the argument is interpreted as an nd-index into the array.

        Returns
        ---
        val : number or array
            Value represented in original format (not binary) casted according to `dtype`.

        """

        if dtype is None:
            dtype = self.vdtype

        if self.val is not None:
            if index is not None:
                raw_val = self.val[index]
            elif item is not None:
                raw_val = self.val.item(item)
            else:
                raw_val = self.val

            if dtype is None:
                val = raw_val / self._get_conv_factor()
            elif dtype == float or np.issubdtype(dtype, np.floating):
                val = raw_val / self._get_conv_factor()
                if isinstance(val, np.ndarray):
                    val = val.astype(dtype)
            elif dtype == int or dtype == 'uint' or dtype == 'int' or np.issubdtype(dtype, np.integer):
                if self.n_frac == 0:
                    val = raw_val
                else:
                    val = raw_val // self._get_conv_factor()
                    val = np.array(list(map(int, val.flatten()))).reshape(val.shape)
                
            elif dtype == complex or np.issubdtype(dtype, np.complexfloating):
                val = (raw_val.real + 1j * raw_val.imag) / self._get_conv_factor()
            else:
                val = raw_val / self._get_conv_factor()
        else:
            val = None

        # scaling reconversion
        if val is not None and self.scaled:
            val = val * self.scale + self.bias
        return val

    def get_val(self, dtype=None, index=None, item=None):
        """
        Returns non-fixed-point value cast to a specified type.

        Paramters
        ---

        dtype : str or dtype, optional, default=None
            Typecode or data-type to which the array is cast.

            `None` returns according to `vdtype` of Fxp, if last one is `None`, `float` is returned.
        
        index : int, optional, default=None
            Index of the element to return from list or array of values cast according `dtype`.

        item : variable number and type, optional, default=None

            None: value is not returned as an item.

            int_type: this argument is interpreted as a flat index into the array, specifying which element to cast and return.

            tuple of int_types: functions as does a single int_type argument, except that the argument is interpreted as an nd-index into the array.

        Returns
        ---
        val : number or array
            Value represented in original format (not binary) casted according to `dtype`.
            
        """

        if dtype is None:
            dtype = self.vdtype
        return self.astype(dtype, index, item)

    def raw(self):
        """
        Returns fixed-point integer value (raw value).

        Notes
        ---

        When Fxp is signed, a signed integer (python `int` or `np.int64` type) is used to represent fixed-point value.
        If a two-complement representation of this integer is needed, use `uraw` method instead.
        """

        return self.val
    
    def uraw(self):
        """
        Returns a tow-complement of fixed-point integer value (unsigned raw value).

        Notes
        ---

        If signed integer representation of this value is needed, use `raw` method instead.
        """
        return np.where(self.val < 0, (1 << self.n_word) + self.val, self.val)

    def equal(self, x, index=None):
        """
        Sets the value of the Fxp using the value of other Fxp object.
        If `x` is not a Fxp, this method set the value just like `set_val` method.

        Parameters
        ---

        x : Fxp object, None, int, float, complex, list of numbers, numpy array, str (bin, hex, dec)
            Value(s) to be stored in fractional fixed-point (base 2) format.

        index : int, optional, default=None
            Index of the element to be overwritten in list or array of values by `val` input.

        Returns
        ---

        self : Fxp object
            Fxp with it's value modified. 
        """
        
        if isinstance(x, Fxp):
            if index is None:
                raw_val = x.val[index]
            else:
                raw_val = x.val

            new_val_raw = raw_val * 2**(self.n_frac - x.n_frac)
            self.set_val(new_val_raw, raw=True, index=index)
        else:
            self.set_val(x, index=index)
        return self

    # behaviors

    def _overflow_action(self, new_val, val_min, val_max):
        if np.any(new_val > val_max):
            self.status['overflow'] = True
            self._run_callbacks('on_status_overflow')
        if np.any(new_val < val_min):
            self.status['underflow'] = True
            self._run_callbacks('on_status_underflow')
        
        if self.config.overflow == 'saturate':
            if isinstance(new_val, np.ndarray) and new_val.dtype == object:
                val = np.clip(new_val, val_min, val_max)
            else:
                val = utils.clip(new_val, val_min, val_max)

        elif self.config.overflow == 'wrap':
            val = utils.wrap(new_val, self.signed, self.n_word)
        else:
            raise ValueError('{} is not a valid config for overflow!'.format(self.config.overflow))
        return val

    def _round(self, val, method='floor'):
        if isinstance(val, int) or np.issubdtype(np.array(val).dtype, np.integer) or np.issubdtype(np.array(val).dtype, np.object_):
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

    def _run_callbacks(self, method):
        if self.callbacks:
            for cb in self.callbacks:
                if hasattr(cb, method): getattr(cb, method)(self)

    # overloadings

    def __call__(self, val=None):
        if val is None:
            rval = self.get_val()
        else:
            rval = self.set_val(val)
        return rval

    def __len__(self):
        return len(self.val)

    def __bool__(self):
        if self.size > 1:
            raise ValueError("The boolean value cannot be determined. Use any() or all().")
        else:
            return bool(self.get_val())

    def __int__(self):
        if self.size > 1:
            raise TypeError('only length-1 arrays can be converted to Python scalars')
        return int(self.astype(int))

    def __float__(self):
        if self.size > 1:
            raise TypeError('only length-1 arrays can be converted to Python scalars')
        return float(self.astype(float))

    def __complex__(self):
        if self.size > 1:
            raise TypeError('only length-1 arrays can be converted to Python scalars')
        return complex(self.astype(complex))
    
    # representation
    
    def __repr__(self):
        dtype_str = self.dtype
        data_str = str(self.get_val()).replace('\n', '\n '+' '*(len(dtype_str)))

        return '{}({})'.format(self.dtype, data_str)

    def __str__(self):
        return str(self.get_val())

    # numpy array representation - numpy hooks
    
    def __array__(self, *args, **kwargs):
        if self.config.array_op_method == 'raw':
            return np.asarray(self.val, *args, **kwargs)
        else:
            return np.asarray(self.get_val(), *args, **kwargs)
    
    def __array_wrap__(self, out_arr, context=None):
        raw = True if self.config.array_op_method == 'raw' else False

        if self.config.array_output_type == 'fxp':
            if self.config.array_op_out is not None:
                return self.config.array_op_out.set_val(out_arr, raw=raw)
            elif self.config.array_op_out_like is not None:
                return self.__class__(out_arr, like=self.config.array_op_out_like, raw=raw)
            else:
                return self.__class__(out_arr)
        else:
            return out_arr

    def __array_prepare__(self, context=None, *args, **kwargs):
        if self.config.array_op_method == 'raw':
            return np.asarray(self.val, *args, **kwargs)
        else:
            return np.asarray(self.get_val(), *args, **kwargs)

    def __array_finalize__(self, obj):
        return
  
    # math operations
    
    def __neg__(self):
        y = Fxp(-self.val, signed=self.signed, n_word=self.n_word, n_frac=self.n_frac, raw=True)
        return y

    def __pos__(self):
        y = Fxp(+self.val, signed=self.signed, n_word=self.n_word, n_frac=self.n_frac, raw=True)
        return y

    def __abs__(self):
        y = Fxp(abs(self.val), signed=self.signed, n_word=self.n_word, n_frac=self.n_frac, raw=True)
        return y          

    def __add__(self, x):
        from .functions import add
        
        if not isinstance(x, Fxp):
            x = self._convert_op_input_value(x)
            _sizing = self.config.const_op_sizing
        else:
            _sizing = self.config.op_sizing

        return add(self, x, out=self.config.op_out, out_like=self.config.op_out_like, sizing=_sizing, method=self.config.op_method)

    __radd__ = __add__

    __iadd__ = __add__

    def __sub__(self, x):
        from .functions import sub

        if not isinstance(x, Fxp):
            x = self._convert_op_input_value(x)
            _sizing = self.config.const_op_sizing
        else:
            _sizing = self.config.op_sizing

        return sub(self, x, out=self.config.op_out, out_like=self.config.op_out_like, sizing=_sizing, method=self.config.op_method)

    def __rsub__(self, x):
        from .functions import sub

        if not isinstance(x, Fxp):
            x = self._convert_op_input_value(x)
            _sizing = self.config.const_op_sizing
            # _sizing = self.config.const_op_sizing if self.config.const_op_sizing != 'same' else 'same_y'
        else:
            _sizing = self.config.op_sizing

        return sub(x, self, out=self.config.op_out, out_like=self.config.op_out_like, sizing=_sizing, method=self.config.op_method)

    __isub__ = __sub__

    def __mul__(self, x):
        from .functions import mul

        if not isinstance(x, Fxp):
            x = self._convert_op_input_value(x)
            _sizing = self.config.const_op_sizing
        else:
            _sizing = self.config.op_sizing

        return mul(self, x, out=self.config.op_out, out_like=self.config.op_out_like, sizing=_sizing, method=self.config.op_method)

    __rmul__ = __mul__

    __imul__ = __mul__

    def __truediv__(self, x):
        from .functions import truediv

        if not isinstance(x, Fxp):
            x = self._convert_op_input_value(x)
            _sizing = self.config.const_op_sizing
        else:
            _sizing = self.config.op_sizing

        return truediv(self, x, out=self.config.op_out, out_like=self.config.op_out_like, sizing=_sizing, method=self.config.op_method)

    def __rtruediv__(self, x):
        from .functions import truediv

        if not isinstance(x, Fxp):
            x = self._convert_op_input_value(x)
            _sizing = self.config.const_op_sizing
        else:
            _sizing = self.config.op_sizing

        return truediv(x, self, out=self.config.op_out, out_like=self.config.op_out_like, sizing=_sizing, method=self.config.op_method)

    __itruediv__ = __truediv__

    def __floordiv__(self, x):
        from .functions import floordiv

        if not isinstance(x, Fxp):
            x = self._convert_op_input_value(x)
            _sizing = self.config.const_op_sizing
        else:
            _sizing = self.config.op_sizing

        return floordiv(self, x, out=self.config.op_out, out_like=self.config.op_out_like, sizing=_sizing, method=self.config.op_method)

    def __rfloordiv__(self, x):
        from .functions import floordiv

        if not isinstance(x, Fxp):
            x = self._convert_op_input_value(x)
            _sizing = self.config.const_op_sizing
        else:
            _sizing = self.config.op_sizing

        return floordiv(x, self, out=self.config.op_out, out_like=self.config.op_out_like, sizing=_sizing, method=self.config.op_method)

    __ifloordiv__ = __floordiv__

    def __mod__(self, x):
        from .functions import mod

        if not isinstance(x, Fxp):
            x = self._convert_op_input_value(x)
            _sizing = self.config.const_op_sizing
        else:
            _sizing = self.config.op_sizing

        return mod(self, x, out=self.config.op_out, out_like=self.config.op_out_like, sizing=_sizing, method=self.config.op_method)

    def __rmod__(self, x):
        from .functions import mod

        if not isinstance(x, Fxp):
            x = self._convert_op_input_value(x)
            _sizing = self.config.const_op_sizing
        else:
            _sizing = self.config.op_sizing

        return mod(x, self, out=self.config.op_out, out_like=self.config.op_out_like, sizing=_sizing, method=self.config.op_method)

    __imod__ = __mod__

    def __pow__(self, x):
        from .functions import pow

        if not isinstance(x, Fxp):
            if self.config is not None and self.config.op_input_size != 'best':
                print("Warning: using config.op_input_size != 'best' could lead to long execution times and huge memory usage! Forcing to config.op_input_size='best'")
                print(f"Tip: force a explicit Fxp dtype for your exponent. Instead of x**{x} use x**Fxp({x}) or x**Fxp({x}, dtype='some fxp dtype')")
            x = self._convert_op_input_value(x, op_input_size='best')
            _sizing = self.config.const_op_sizing
        else:
            _sizing = self.config.op_sizing

        return pow(self, x, out=self.config.op_out, out_like=self.config.op_out_like, sizing=_sizing, method=self.config.op_method)

    __rpow__ = __pow__

    __ipow__ = __pow__

    
    # bit level operators

    def __rshift__(self, n):
        if self.config.shifting == 'expand':
            min_pow2 = utils.min_pow2(self.val)     # minimum power of 2 in raw val
            if min_pow2 is not None and n > min_pow2:
                n_frac_expansion = n - min_pow2
            else:
                n_frac_expansion = 0
            
            y = Fxp(None, signed=self.signed, n_word=self.n_word+n_frac_expansion, n_frac=self.n_frac+n_frac_expansion)
            y.set_val(self.val >> np.array(n - n_frac_expansion, dtype=self.val.dtype), raw=True)   # set raw val shifted
        else:
            y = self.deepcopy()
            y.val = y.val >> np.array(n, dtype=y.val.dtype)
        return y

    __irshift__ = __rshift__

    def __lshift__(self, n):
        if self.config.shifting == 'expand':
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
        # return Fxp(self.val[index], like=self, raw=True)
        y = Fxp(like=self)
        y.val = self.val[index]
        return y

    def __setitem__(self, index, value):
        self.set_val(value, index=index)

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
            s += '\tOverflow\t=\t{}\n'.format(self.config.overflow)
            s += '\tRounding\t=\t{}\n'.format(self.config.rounding)
            s += '\tShifting\t=\t{}\n'.format(self.config.shifting)
        print(s)


    # base representations
    def bin(self, frac_dot=False, prefix=None):
        if frac_dot:
            n_frac_dot = self.n_frac
        else:
            n_frac_dot = None

        # set prefix if it's necessary
        prefix = prefix if prefix is not None else self.config.bin_prefix
        if prefix is not None:
            if isinstance(prefix, bool) and prefix == True:
                prefix = '0b' # default binary prefix
        
        if isinstance(self.val, (list, np.ndarray)) and self.val.ndim > 0:
            if self.vdtype == complex:
                real_val = [utils.binary_repr(utils.int_array(val.real), n_word=self.n_word, n_frac=n_frac_dot, prefix=prefix) for val in self.val]
                imag_val = [utils.binary_repr(utils.int_array(val.imag), n_word=self.n_word, n_frac=n_frac_dot, prefix=prefix) for val in self.val]
                rval = utils.complex_repr(real_val, imag_val)
            else:
                rval = [utils.binary_repr(utils.int_array(val), n_word=self.n_word, n_frac=n_frac_dot, prefix=prefix) for val in self.val]
        else:
            if self.vdtype == complex:
                real_val = utils.binary_repr(utils.int_array(self.val.real), n_word=self.n_word, n_frac=n_frac_dot, prefix=prefix)
                imag_val = utils.binary_repr(utils.int_array(self.val.imag), n_word=self.n_word, n_frac=n_frac_dot, prefix=prefix)
                rval = utils.complex_repr(real_val, imag_val)
            else:
                rval = utils.binary_repr(int(self.val), n_word=self.n_word, n_frac=n_frac_dot, prefix=prefix)

        return rval

    def hex(self, padding=True, prefix=None):
        if padding:
            hex_n_word = self.n_word
        else:
            hex_n_word = None

        # set prefix if it's necessary
        prefix = prefix if prefix is not None else self.config.hex_prefix
        if prefix is not None:
            if isinstance(prefix, bool) and prefix == True:
                prefix = '0x' # default hexadecimal prefix

        if isinstance(self.val, (list, np.ndarray)) and self.val.ndim > 0:
            if self.vdtype == complex:
                real_val = [utils.hex_repr(utils.binary_repr(utils.int_array(val.real), n_word=self.n_word, n_frac=None), n_word=hex_n_word, base=2, prefix=prefix) for val in self.val]
                imag_val = [utils.hex_repr(utils.binary_repr(utils.int_array(val.imag), n_word=self.n_word, n_frac=None), n_word=hex_n_word, base=2, prefix=prefix) for val in self.val]
                rval = utils.complex_repr(real_val, imag_val)
            else:
                rval = [utils.hex_repr(val, n_word=hex_n_word, base=2, prefix=prefix) for val in self.bin()]
        else:
            if self.vdtype == complex:
                real_val = utils.hex_repr(utils.binary_repr(utils.int_array(self.val.real), n_word=self.n_word, n_frac=None), n_word=hex_n_word, base=2, prefix=prefix)
                imag_val = utils.hex_repr(utils.binary_repr(utils.int_array(self.val.imag), n_word=self.n_word, n_frac=None), n_word=hex_n_word, base=2, prefix=prefix)
                rval = utils.complex_repr(real_val, imag_val)
            else:
                rval = utils.hex_repr(self.bin(), n_word=hex_n_word, base=2, prefix=prefix)
        return rval
    
    def base_repr(self, base, frac_dot=False):
        if frac_dot:
            n_frac_dot = self.n_frac
        else:
            n_frac_dot = None

        if isinstance(self.val, (list, np.ndarray)) and self.val.ndim > 0:
            if self.vdtype == complex:
                real_val = [utils.base_repr(utils.int_array(val.real), base=base, n_frac=n_frac_dot)  for val in self.val]
                imag_val = [utils.base_repr(utils.int_array(val.imag), base=base, n_frac=n_frac_dot)  for val in self.val]
                rval = utils.complex_repr(real_val, imag_val)
            else:
                rval = [utils.base_repr(utils.int_array(val), base=base, n_frac=n_frac_dot) for val in self.val]
        else:
            if self.vdtype == complex:
                rval = utils.complex_repr(utils.base_repr(int(self.val.real), base=base, n_frac=n_frac_dot), utils.base_repr(int(self.val.imag), base=base, n_frac=n_frac_dot))
            else:
                rval = utils.base_repr(int(self.val), base=base, n_frac=n_frac_dot)
        return rval

    def from_bin(self, val, raw=False):
        self.set_val(utils.add_binary_prefix(val), raw=raw)
        return self

    # copy
    def copy(self):
        return copy.copy(self)

    def deepcopy(self):
        return copy.deepcopy(self)

    def like(self, x):
        if isinstance(x, self.__class__):
            new_raw_val = self.val * 2**(x.n_frac - self.n_frac)
            return  x.copy().set_val(new_raw_val, raw=True)
        else:
            raise ValueError('`x` should be a Fxp object!')

    # reset
    def reset(self):
        #status (overwrite)
        self.status = {
            'overflow': False,
            'underflow': False,
            'inaccuracy': False}

    def _convert_op_input_value(self, x, op_input_size=None):
        if not isinstance(x, Fxp):
            if op_input_size is None and self.config is not None:
                op_input_size = self.config.op_input_size

            if op_input_size is None:
                x_fxp = Fxp(x)
            elif op_input_size == 'best':
                x_fxp = Fxp(x)
            elif op_input_size == 'same':
                x_fxp = Fxp(x, like=self)
            else:
                raise ValueError('Sizing parameter not supported: {}'.format(op_input_size))
        else:
            x_fxp = x

        return x_fxp

    # endregion

    # numpy functions dispatch
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == '__call__':
            if ufunc in _NUMPY_HANDLED_FUNCTIONS:
                # dispatch function to implemented in fxpmath
                return self._set_array_output_type(_NUMPY_HANDLED_FUNCTIONS[ufunc](*inputs, **kwargs))

            # call original numpy function and return wrapped result
            kwargs['method'] = method
            return self._wrapped_numpy_func(ufunc, *inputs, **kwargs)
        else:
            # return NotImplemented

            # call original numpy function and return wrapped result
            kwargs['method'] = method
            return self._wrapped_numpy_func(ufunc, *inputs, **kwargs)

    def __array_function__(self, func, types, args, kwargs):
        if func not in _NUMPY_HANDLED_FUNCTIONS:
            # return NotImplemented

            # call original numpy function and return wrapped result
            return self._wrapped_numpy_func(func, *args, **kwargs)

        # Note: this allows subclasses that don't override
        # __array_function__ to handle Fxp objects
        if not all(issubclass(t, self.__class__) for t in types):
            # return NotImplemented
            pass    # delegates to implemented functions deal with conversion

        # dispatch function to implemented in fxpmath
        return self._set_array_output_type(_NUMPY_HANDLED_FUNCTIONS[func](*args, **kwargs))

    def _wrapped_numpy_func(self, func, *args, **kwargs):
        # convert func inputs to numpy arrays
        args = [np.asarray(arg) if isinstance(arg, self.__class__) else arg for arg in args]

        # out parameter extraction if Fxp
        out = None
        if 'out' in kwargs:
            if isinstance(kwargs['out'], self.__class__):
                out = kwargs.pop('out', None)
            elif (isinstance(kwargs['out'], tuple) and isinstance(kwargs['out'][0], self.__class__)):
                out = kwargs.pop('out', None)[0]
            else:
                out = None
                kwargs.pop('out')

        # out parameter extraction if Fxp
        out_like = None
        if 'out_like' in kwargs:
            if isinstance(kwargs['out_like'], self.__class__):
                out_like = kwargs.pop('out_like', None)
            elif (isinstance(kwargs['out_like'], tuple) and isinstance(kwargs['out_like'][0], self.__class__)):
                out_like = kwargs.pop('out_like', None)
            else:
                out_like = None
                kwargs.pop('out_like')

        # get function if a method is specified
        if 'method' in kwargs  and isinstance (kwargs['method'], str):
            method = kwargs.pop('method')
            func = getattr(func, method)
            
        # calculate (call original numpy function)
        try:
            val = func(*args, **kwargs)
        except TypeError:
            # call function converting args to float type (this is because numpy issue about pass object type to ufunc)
            # args = [arg.astype(float) if isinstance(arg, np.ndarray) else arg for arg in args]

            args_converted = []
            for arg in args:
                if isinstance(arg, np.ndarray):
                    if isinstance(arg.item(0), complex):
                        args_converted.append(arg.astype(complex))
                    else:
                        args_converted.append(arg.astype(float))
                else:
                    args_converted.append(arg)
            # call func
            val = func(*args_converted, **kwargs)

        if out is not None:
            return out(val)
        elif out_like is not None:
            return self.__class__(val, like=out_like)
        else:
            # return wrapped result
            return self.__array_wrap__(val)

    def _set_array_output_type(self, out_arr):
        if self.config._array_output_type == 'fxp':
            raw = True if self.config.array_op_method == 'raw' else False

            if self.config.array_op_out is not None:
                return self.config.array_op_out.set_val(out_arr, raw=raw)
            elif self.config.array_op_out_like is not None:
                return self.__class__(out_arr, like=self.config.array_op_out_like, raw=raw)
            elif not isinstance(out_arr, self.__class__):
                return self.__class__(out_arr)

        elif self.config._array_output_type == 'array' and isinstance(out_arr, self.__class__):
            return np.asarray(out_arr.get_val())
        
        return out_arr


    # methods derived from Numpy ndarray

    def all(self, axis=None, **kwargs):
        return np.all(np.array(self), axis=axis, **kwargs)

    def any(self, axis=None, **kwargs):
        return np.any(np.array(self), axis=axis, **kwargs)

    def argmax(self, axis=None, **kwargs):
        return np.argmax(self.val, axis=axis, **kwargs)    # operates over raw values

    def argmin(self, axis=None, **kwargs):
        return np.argmin(self.val, axis=axis, **kwargs)    # operates over raw values

    def argpartition(self, axis=-1, **kwargs):
        return np.argpartition(self.val, axis=axis, **kwargs)    # operates over raw values

    def argsort(self, axis=-1, **kwargs):
        return np.argsort(self.val, axis=axis, **kwargs)    # operates over raw values

    def nonzero(self):
        from .functions import nonzero
        return nonzero(self)  

    def max(self, axis=None, **kwargs):
        from .functions import fxp_max

        out = kwargs.pop('out', self.config.op_out)
        out_like = kwargs.pop('out_like', self.config.op_out_like)
        sizing = kwargs.pop('sizing', self.config.op_sizing)
        method = kwargs.pop('method', self.config.op_method)

        return fxp_max(self, axis=axis, out=out, out_like=out_like, sizing=sizing, method=method, **kwargs)

    def min(self, axis=None, **kwargs):
        from .functions import fxp_min

        out = kwargs.pop('out', self.config.op_out)
        out_like = kwargs.pop('out_like', self.config.op_out_like)
        sizing = kwargs.pop('sizing', self.config.op_sizing)
        method = kwargs.pop('method', self.config.op_method)

        return fxp_min(self, axis=axis, out=out, out_like=out_like, sizing=sizing, method=method, **kwargs)

    def mean(self, axis=None, **kwargs):
        if not 'out' in kwargs and \
            self.config.array_output_type == 'fxp' and \
            self.config.array_op_out is None and self.config.array_op_out_like is None: 
            
            kwargs['out'] = Fxp(like=self)  # define Fxp output with same size by default

        return np.mean(self, axis=axis, **kwargs)

    def std(self, axis=None, **kwargs):
        if not 'out' in kwargs and \
            self.config.array_output_type == 'fxp' and \
            self.config.array_op_out is None and self.config.array_op_out_like is None: 
            
            kwargs['out'] = Fxp(like=self)  # define Fxp output with same size by default

        return np.std(self, axis=axis, **kwargs)

    def var(self, axis=None, **kwargs):
        if not 'out' in kwargs and \
            self.config.array_output_type == 'fxp' and \
            self.config.array_op_out is None and self.config.array_op_out_like is None: 
            
            kwargs['out'] = Fxp(like=self)  # define Fxp output with same size by default

        return np.var(self, axis=axis, **kwargs)

    def sum(self, axis=None, **kwargs):
        from .functions import sum

        out = kwargs.pop('out', self.config.op_out)
        out_like = kwargs.pop('out_like', self.config.op_out_like)
        sizing = kwargs.pop('sizing', self.config.op_sizing)
        method = kwargs.pop('method', self.config.op_method)

        return sum(self, axis=axis, out=out, out_like=out_like, sizing=sizing, method=method, **kwargs)

    def cumsum(self, axis=None, **kwargs):
        from .functions import cumsum

        out = kwargs.pop('out', self.config.op_out)
        out_like = kwargs.pop('out_like', self.config.op_out_like)
        sizing = kwargs.pop('sizing', self.config.op_sizing)
        method = kwargs.pop('method', self.config.op_method)

        return cumsum(self, axis=axis, out=out, out_like=out_like, sizing=sizing, method=method, **kwargs)

    def cumprod(self, axis=None, **kwargs):
        from .functions import cumprod

        out = kwargs.pop('out', self.config.op_out)
        out_like = kwargs.pop('out_like', self.config.op_out_like)
        sizing = kwargs.pop('sizing', self.config.op_sizing)
        method = kwargs.pop('method', self.config.op_method)

        return cumprod(self, axis=axis, out=out, out_like=out_like, sizing=sizing, method=method, **kwargs)

    ravel = flatten

    def tolist(self):
        return self.get_val().tolist()

    def sort(self, axis=-1, **kwargs):
        self.val.sort(axis=axis, **kwargs)

    def conjugate(self, **kwargs):
        from .functions import conjugate

        out = kwargs.pop('out', self.config.op_out)
        out_like = kwargs.pop('out_like', self.config.op_out_like)
        sizing = kwargs.pop('sizing', self.config.op_sizing)
        method = kwargs.pop('method', self.config.op_method)

        return conjugate(self, out=out, out_like=out_like, sizing=sizing, method=method, **kwargs)

    conj = conjugate

    @property
    def T(self):
        x = self.copy()
        x.val = x.val.T
        return x    
    
    def transpose(self, axes=None, **kwargs):
        from .functions import transpose

        out = kwargs.pop('out', self.config.op_out)
        out_like = kwargs.pop('out_like', self.config.op_out_like)
        sizing = kwargs.pop('sizing', self.config.op_sizing)
        method = kwargs.pop('method', self.config.op_method)

        return transpose(self, axes=axes, out=out, out_like=out_like, sizing=sizing, method=method, **kwargs)

    def item(self, *args):
        if len(args) > 1:
            items = tuple(args)
        else:
            items = args[0]
        return self.astype(item=items)

    def clip(self, a_min=None, a_max=None, **kwargs):
        from .functions import clip

        out = kwargs.pop('out', self.config.op_out)
        out_like = kwargs.pop('out_like', self.config.op_out_like)
        sizing = kwargs.pop('sizing', self.config.op_sizing)
        method = kwargs.pop('method', self.config.op_method)

        return clip(self, a_min=a_min, a_max=a_max, out=out, out_like=out_like, sizing=sizing, method=method, **kwargs)

    def diagonal(self, offset=0, axis1=0, axis2=1, **kwargs):
        from .functions import diagonal

        out = kwargs.pop('out', self.config.op_out)
        out_like = kwargs.pop('out_like', self.config.op_out_like)
        sizing = kwargs.pop('sizing', self.config.op_sizing)
        method = kwargs.pop('method', self.config.op_method)

        return diagonal(self, offset=offset, axis1=axis1, axis2=axis2, out=out, out_like=out_like, sizing=sizing, method=method, **kwargs)  

    def trace(self, offset=0, axis1=0, axis2=1, **kwargs):
        from .functions import trace

        out = kwargs.pop('out', self.config.op_out)
        out_like = kwargs.pop('out_like', self.config.op_out_like)
        sizing = kwargs.pop('sizing', self.config.op_sizing)
        method = kwargs.pop('method', self.config.op_method)

        return trace(self, offset=offset, axis1=axis1, axis2=axis2, out=out, out_like=out_like, sizing=sizing, method=method, **kwargs) 

    def prod(self, axis=None, **kwargs):
        from .functions import prod

        out = kwargs.pop('out', self.config.op_out)
        out_like = kwargs.pop('out_like', self.config.op_out_like)
        sizing = kwargs.pop('sizing', self.config.op_sizing)
        method = kwargs.pop('method', self.config.op_method)

        return prod(self, axis=axis, out=out, out_like=out_like, sizing=sizing, method=method, **kwargs) 

    def dot(self, x, **kwargs):
        from .functions import dot

        if not isinstance(x, Fxp):
            x = self._convert_op_input_value(x)
            _sizing = self.config.const_op_sizing
        else:
            _sizing = self.config.op_sizing

        out = kwargs.pop('out', self.config.op_out)
        out_like = kwargs.pop('out_like', self.config.op_out_like)
        sizing = kwargs.pop('sizing', _sizing)
        method = kwargs.pop('method', self.config.op_method)

        return dot(self, x, out=out, out_like=out_like, sizing=sizing, method=method, **kwargs) 

class Config():
    template = None

    def __init__(self, **kwargs):
        # size limits
        self.max_error = kwargs.pop('max_error', 1 / 2**63)
        self.n_word_max = kwargs.pop('n_word_max', 64)

        # behavior
        self.overflow = kwargs.pop('overflow', 'saturate')
        self.rounding = kwargs.pop('rounding', 'trunc')
        self.shifting = kwargs.pop('shifting', 'expand')
        self.op_method = kwargs.pop('op_method', 'raw')

        # inputs
        self.op_input_size = kwargs.pop('op_input_size', 'same')

        # alu ops outpus
        self.op_out = kwargs.pop('op_out', None)
        self.op_out_like = kwargs.pop('op_out_like', None)
        self.op_sizing = kwargs.pop('op_sizing', 'optimal')

        # alu ops with a constant operand
        self.const_op_sizing = kwargs.pop('const_op_sizing', 'same')

        # array ops
        self.array_output_type = kwargs.pop('array_output_type', 'fxp')
        self.array_op_out = kwargs.pop('array_op_out', None)
        self.array_op_out_like = kwargs.pop('array_op_out_like', None)
        self.array_op_method = kwargs.pop('array_op_method', 'repr')

        # notation
        self.dtype_notation = kwargs.pop('dtype_notation', 'fxp')

        # update from template
        # if `template` is in kwarg, the reference template is updated
        if 'template' in kwargs: self.template = kwargs.pop('template')

        if self.template is not None:
            if isinstance(self.template, Config):
                self.__dict__ = copy.deepcopy(self.template.__dict__)

        # prefixes
        self.bin_prefix = kwargs.pop('bin_prefix', None)
        self.hex_prefix = kwargs.pop('hex_prefix', '0x')

    # ---
    # properties
    # ---
    # region

    # max_error
    @property
    def max_error(self):
        return self._max_error
    
    @max_error.setter
    def max_error(self, val):
        if val > 0:
            self._max_error = val
        else:
            raise ValueError('max_error must be greater than 0!')

    # n_word_max
    @property
    def n_word_max(self):
        return self._n_word_max
    
    @n_word_max.setter
    def n_word_max(self, val):
        if isinstance(val, int) and val > 0:
            self._n_word_max = val
        else:
            raise ValueError('n_word_max must be int type greater than 0!')

    # overflow
    @property
    def _overflow_list(self):
        return ['saturate', 'wrap']

    @property
    def overflow(self):
        return self._overflow
    
    @overflow.setter
    def overflow(self, val):
        if isinstance(val, str) and val in self._overflow_list:
            self._overflow = val
        else:
            raise ValueError('overflow must be str type with following valid values: {}'.format(self._overflow_list))

    # rounding
    @property
    def _rounding_list(self):
        return ['around', 'floor', 'ceil', 'fix', 'trunc']

    @property
    def rounding(self):
        return self._rounding
    
    @rounding.setter
    def rounding(self, val):
        if isinstance(val, str) and val in self._rounding_list:
            self._rounding = val
        else:
            raise ValueError('rounding must be str type with following valid values: {}'.format(self._rounding_list))

    # shifting
    @property
    def _shifting_list(self):
        return ['expand', 'trunc', 'keep']

    @property
    def shifting(self):
        return self._shifting
    
    @shifting.setter
    def shifting(self, val):
        if isinstance(val, str) and val in self._shifting_list:
            self._shifting = val
        else:
            raise ValueError('shifting must be str type with following valid values: {}'.format(self._shifting_list))

    # op_input_size
    @property
    def _op_input_size_list(self):
        return ['same', 'best']

    @property
    def op_input_size(self):
        return self._op_input_size
    
    @op_input_size.setter
    def op_input_size(self, val):
        if isinstance(val, str) and val in self._op_input_size_list:
            self._op_input_size = val
        else:
            raise ValueError('op_input_size must be str type with following valid values: {}'.format(self._op_input_size_list))
    

    # op_out
    @property
    def op_out(self):
        return self._op_out
    
    @op_out.setter
    def op_out(self, val):
        if val is None or isinstance(val, Fxp):
            self._op_out = val
        else:
            raise ValueError('op_out must be a Fxp object or None!')

    # op_out_like
    @property
    def op_out_like(self):
        return self._op_out_like
    
    @op_out_like.setter
    def op_out_like(self, val):
        if val is None or isinstance(val, Fxp):
            self._op_out_like = val
        else:
            raise ValueError('op_out_like must be a Fxp object or None!')

    # op_sizing
    @property
    def _op_sizing_list(self):
        return ['optimal', 'same', 'fit', 'largest', 'smallest']

    @property
    def op_sizing(self):
        return self._op_sizing
    
    @op_sizing.setter
    def op_sizing(self, val):
        if isinstance(val, str) and val in self._op_sizing_list:
            self._op_sizing = val
        else:
            raise ValueError('op_sizing must be str type with following valid values: {}'.format(self._op_sizing_list))

    # op_method
    @property
    def _op_method_list(self):
        return ['raw', 'repr']

    @property
    def op_method(self):
        return self._op_method
    
    @op_method.setter
    def op_method(self, val):
        if isinstance(val, str) and val in self._op_method_list:
            self._op_method = val
        else:
            raise ValueError('op_method must be str type with following valid values: {}'.format(self._op_method_list))

    # const_op_sizing
    @property
    def _const_op_sizing_list(self):
        return ['optimal', 'same', 'fit', 'largest', 'smallest']

    @property
    def const_op_sizing(self):
        return self._const_op_sizing
    
    @const_op_sizing.setter
    def const_op_sizing(self, val):
        if isinstance(val, str) and val in self._const_op_sizing_list:
            self._const_op_sizing = val
        else:
            raise ValueError('op_sizing must be str type with following valid values: {}'.format(self._const_op_sizing_list))

    # array_output_type
    @property
    def _array_output_type_list(self):
        return ['fxp', 'array']

    @property
    def array_output_type(self):
        return self._array_output_type
    
    @array_output_type.setter
    def array_output_type(self, val):
        if isinstance(val, str) and val in self._array_output_type_list:
            self._array_output_type = val
        else:
            raise ValueError('array_output_type must be str type with following valid values: {}'.format(self._array_output_type_list))

    # array_op_out
    @property
    def array_op_out(self):
        return self._array_op_out
    
    @array_op_out.setter
    def array_op_out(self, val):
        if val is None or isinstance(val, Fxp):
            self._array_op_out = val
        else:
            raise ValueError('array_op_out must be a Fxp object or None!')

    # array_op_out_like
    @property
    def array_op_out_like(self):
        return self._array_op_out_like
    
    @array_op_out_like.setter
    def array_op_out_like(self, val):
        if val is None or isinstance(val, Fxp):
            self._array_op_out_like = val
        else:
            raise ValueError('array_op_out_like must be a Fxp object or None!')

    # array_op_method
    @property
    def _array_op_method_list(self):
        return ['raw', 'repr']

    @property
    def array_op_method(self):
        return self._array_op_method
    
    @array_op_method.setter
    def array_op_method(self, val):
        if isinstance(val, str) and val in self._array_op_method_list:
            self._array_op_method = val
        else:
            raise ValueError('array_op_method must be str type with following valid values: {}'.format(self._array_op_method_list))

    # dtype_notation
    @property
    def _dtype_notation_list(self):
        return ['fxp', 'Q']

    @property
    def dtype_notation(self):
        return self._dtype_notation
    
    @dtype_notation.setter
    def dtype_notation(self, val):
        if isinstance(val, str) and val in self._dtype_notation_list:
            self._dtype_notation = val
        else:
            raise ValueError('dtype_notation must be str type with following valid values: {}'.format(self._dtype_notation_list))

    # prefixes
    @property
    def bin_prefix(self):
        return self._bin_prefix
    
    @bin_prefix.setter
    def bin_prefix(self, prefix):
        if prefix is not None and not isinstance(prefix, str):
            print("Warning: the prefix should be a string, converted to string automatically!")
            prefix = str(prefix)

        if prefix not in [None, 'b', '0b', 'B', '0B']:
            print(f"Warning: the prefix {prefix} is not a common prefix for binary values!")

        self._bin_prefix = prefix

    @property
    def hex_prefix(self):
        return self._hex_prefix
    
    @hex_prefix.setter
    def hex_prefix(self, prefix):
        if prefix is not None and not isinstance(prefix, str):
            print("Warning: the prefix should be a string, converted to string automatically!")
            prefix = str(prefix)

        if prefix not in [None, 'x', '0x', 'X', '0X', 'h', '0h', 'H', '0H']:
            print(f"Warning: the prefix {prefix} is not a common prefix for hexadecimal values!")

        self._hex_prefix = prefix

    # endregion

    # ---
    # methods
    # ---
    # region

    def print(self):
        for k, v in self.__dict__.items():
            print('\t{: <24}:\t{}'.format(k.strip('_'), v))

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

    # copy
    def copy(self):
        return copy.copy(self)

    def deepcopy(self):
        return copy.deepcopy(self)
    # endregion

# ----------------------------------------------------------------------------------------
# Internal functions
# ----------------------------------------------------------------------------------------
def implements(*np_functions):
   "Register an __array_function__ implementation for Fxp objects."
   def decorator(fxp_func):
        for np_func in np_functions:
            _NUMPY_HANDLED_FUNCTIONS[np_func] = fxp_func
        return fxp_func
   return decorator
