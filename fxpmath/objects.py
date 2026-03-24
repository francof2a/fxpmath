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
import math
import copy
import re
import warnings

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
    """Numerical Fractional Fixed-Point object (base 2).
    
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
    
    Examples
    ---
    >>> from fxpmath import Fxp
    >>> x = Fxp(3.75, signed=True, n_word=8, n_frac=4)
    >>> x()
    3.75
    >>> x.bin(frac_dot=True)
    '0b0011.1100'"""

    template = None

    def __init__(self, val=None, signed=None, n_word=None, n_frac=None, n_int=None, like=None, dtype=None, **kwargs):

        # Init all properties in None
        """Initialize the instance.
        
        Parameters
        ---
        val : scalar, complex, array_like, or Fxp, optional
            Value(s) used to initialize, convert, or assign fixed-point data.
        signed : bool, optional
            Whether the fixed-point format includes a sign bit (`True`) or is unsigned (`False`).
        n_word : int, optional
            Total word length in bits.
        n_frac : int, optional
            Number of fractional bits.
        n_int : int, optional
            Number of integer bits (excluding sign bit).
        like : Fxp or None, optional
            Template `Fxp` object used to copy dtype/config defaults.
        dtype : str or None, optional
            Fixed-point dtype string in `fxp-*` or `Q/UQ` notation.
        **kwargs : dict
            Extra keyword arguments propagated to lower-level conversion or NumPy handlers.
        
        Side Effects
        ---
        Initializes value storage, dtype fields, status flags, configuration, and callback bindings."""
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
        """Return the current fixed-point dtype string (for example `fxp-s16/8`)."""
        return self._dtype

    # overflow (mirror of config for compatibility)
    @property
    def overflow(self):
        """Return the selected overflow handling mode."""
        return self.config.overflow
    
    @overflow.setter
    def overflow(self, val):
        """Set `overflow` behavior through the attached `Config` object.
        
        Parameters
        ---
        val : scalar, complex, array_like, or Fxp
            Value(s) used to initialize, convert, or assign fixed-point data.
        
        Side Effects
        ---
        Updates internal fixed-point object state in place."""
        self.config.overflow = val

    # rounding (mirror of config for compatibility)
    @property
    def rounding(self):
        """Return the selected rounding mode."""
        return self.config.rounding
    
    @rounding.setter
    def rounding(self, val):
        """Set `rounding` behavior through the attached `Config` object.
        
        Parameters
        ---
        val : scalar, complex, array_like, or Fxp
            Value(s) used to initialize, convert, or assign fixed-point data.
        
        Side Effects
        ---
        Updates internal fixed-point object state in place."""
        self.config.rounding = val

    # shifting (mirror of config for compatibility)
    @property
    def shifting(self):
        """Return the selected shifting mode."""
        return self.config.shifting
    
    @shifting.setter
    def shifting(self, val):
        """Set `shifting` behavior through the attached `Config` object.
        
        Parameters
        ---
        val : scalar, complex, array_like, or Fxp
            Value(s) used to initialize, convert, or assign fixed-point data.
        
        Side Effects
        ---
        Updates internal fixed-point object state in place."""
        self.config.shifting = val

    @property
    def shape(self):
        """Return the shape of the stored value array."""
        return self.val.shape

    @shape.setter
    def shape(self, shape):
        """Set shape in place with a deprecation warning; prefer `reshape` or `reshape_inplace`."""
        warnings.warn(
            "Assigning to `shape` is deprecated and may be removed in a future release. Use `reshape()` or `reshape_inplace()` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.reshape_inplace(shape=shape)

    @property
    def ndim(self):
        """Return the number of dimensions of the stored value array."""
        return self.val.ndim

    @property
    def size(self):
        """Return the number of stored scalar elements."""
        return self.val.size


    # endregion

    # ---
    # Methods
    # ---
    # region

    # methods about size

    def get_dtype(self, notation=None):
        """Get dtype attribute of Fxp.
        
        Parameters
        ---
        notation : {None, 'fxp', 'Q'}, optional
            Notation used to format dtype strings. `None` uses `config.dtype_notation`; `'Q'` selects `Q/UQ` format; `'fxp'` selects `fxp-s/u` format.
        
        Returns
        ---
        str
            Fixed-point dtype string in the selected notation format."""
        self._update_dtype(notation)    # update dtype
        return self._dtype
    
    def _qfmt(self):
        """Build a Q/UQ notation string from the current fixed-point configuration."""
        return re.compile(r'(s|u|q|uq|qu)(\d+)(\.[+-]?\d+)?')
    
    def _fxpfmt(self):
        """Build an fxp notation string from the current fixed-point configuration."""
        return re.compile(r'fxp-(s|u)(\d+)/([+-]?\d+)(-complex)?')
    
    def _parseformatstr(self, fmt):
        """Parse a format string and return signedness, word size, fractional bits, and complex flag.
        
        Parameters
        ---
        fmt : str
            Format string to parse (supports `Q/UQ` and `fxp-s/u` forms).
        
        Returns
        ---
        tuple[bool, int, int, bool]
            Parsed `(signed, n_word, n_frac, complex_dtype)` tuple."""
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
        """Initialize signedness and bit-width fields from explicit sizes or inferred best-fit sizes.
        
        Parameters
        ---
        val : scalar, complex, array_like, or Fxp, optional
            Value(s) used to initialize, convert, or assign fixed-point data.
        signed : bool, optional
            Whether the fixed-point format includes a sign bit (`True`) or is unsigned (`False`).
        n_word : int, optional
            Total word length in bits.
        n_frac : int, optional
            Number of fractional bits.
        n_int : int, optional
            Number of integer bits (excluding sign bit).
        max_error : float, optional
            Maximum absolute representation error allowed when inferring bit widths.
        n_word_max : int, optional
            Maximum allowed inferred word length.
        raw : bool, optional
            When `True`, treat input values as raw stored integers instead of represented values.
        
        Side Effects
        ---
        Updates sizing attributes (`signed`, `n_word`, `n_int`, `n_frac`) based on explicit or inferred limits."""
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
        """Change size of one or more of size parameters (signed, n_word, n_int and/or n_frac) of the Fxp object.
        
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
        
        Returns
        ---
        None
            This method updates the object in place."""
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

        """Infer fixed-point bit widths that fit the provided value with the configured error tolerance.
        
        Parameters
        ---
        val : scalar, complex, array_like, or Fxp, optional
            Value(s) used to initialize, convert, or assign fixed-point data.
        n_word : int, optional
            Total word length in bits.
        n_frac : int, optional
            Number of fractional bits.
        max_error : float, optional
            Maximum absolute representation error allowed when inferring bit widths.
        n_word_max : int, optional
            Maximum allowed inferred word length.
        raw : bool, optional
            When `True`, treat input values as raw stored integers instead of represented values.
        
        Side Effects
        ---
        Infers best-fit size fields and applies them to this instance."""
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
        """Reshape the fixed-point array.
        
        Parameters
        ---
        shape : int or tuple[int, ...]
            Target shape for array reinterpretation.
        order : {'C', 'F', 'A', 'K'}, optional
            Memory order used by reshape/flatten operations.
        
        Returns
        ---
        Fxp
            New `Fxp` instance with reshaped underlying values and preserved format/configuration."""

        x = self.copy()
        x.reshape_inplace(shape=shape, order=order)
        return x

    def reshape_inplace(self, shape, order='C'):
        """Reshape the fixed-point array in place.

        Parameters
        ---
        shape : int or tuple[int, ...]
            Target shape for array reinterpretation.
        order : {'C', 'F', 'A', 'K'}, optional
            Memory order used by reshape/flatten operations.

        Returns
        ---
        Fxp
            This instance after reshaping its stored values in place."""

        self.val = self.val.reshape(shape, order=order)
        if isinstance(self.real, np.ndarray):
            self.real = self.real.reshape(shape, order=order)
        if isinstance(self.imag, np.ndarray):
            self.imag = self.imag.reshape(shape, order=order)
        return self
    
    def flatten(self, order='C'):
        """Return a copy of the Fxp with its values array collapsed into one dimension.
        
        Parameters
        ---
        order : {'C', 'F', 'A', 'K'}, optional
            Memory order used by reshape/flatten operations.
        
        Returns
        ---
        Fxp
            New one-dimensional `Fxp` copy of the stored values."""

        x = self.copy()
        x.val = x.val.flatten(order)
        return x

    # methods about value

    def _format_inupt_val(self, val, return_sizes=False, raw=False, set_inaccuracy=True):
        """Normalize input values into an internal ndarray representation and derive dtype metadata.
        
        Parameters
        ---
        val : scalar, complex, array_like, or Fxp
            Value(s) used to initialize, convert, or assign fixed-point data.
        return_sizes : bool, optional
            When `True`, also return inferred signed/size metadata.
        raw : bool, optional
            When `True`, treat input values as raw stored integers instead of represented values.
        set_inaccuracy : bool, optional
            Whether quantization differences should set the inaccuracy status flag.
        
        Returns
        ---
        tuple
            Normalized value plus inferred dtype metadata used during assignment."""
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

        """Return the scale factor used to convert between raw integers and represented values.
        
        Parameters
        ---
        raw : bool, optional
            When `True`, treat input values as raw stored integers instead of represented values.
        
        Returns
        ---
        int, float, or complex
            Scale factor used to convert between raw storage and represented values."""
        if raw:
            conv_factor = 1
        elif self.n_frac >= 0:
            conv_factor = (1<<self.n_frac)
        else:
            conv_factor = 1/(1<<-self.n_frac)

        return conv_factor

    def _update_dtype(self, notation=None):
        """Rebuild the dtype string based on current sizing and notation configuration.
        
        Parameters
        ---
        notation : {None, 'fxp', 'Q'}, optional
            Notation used to format dtype strings. `None` uses `config.dtype_notation`; `'Q'` selects `Q/UQ` format; `'fxp'` selects `fxp-s/u` format.
        
        Side Effects
        ---
        Rebuilds and stores the internal dtype string (`self._dtype`)."""
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
        """Set the value/s of the Fxp object.
        
        Parameters
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
            Fxp with it's value modified."""

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
                new_val = utils.int_array(new_val).astype(val_dtype)
            
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
                new_val_real = utils.int_array(new_val_real)
                new_val_imag = utils.int_array(new_val_imag)
            
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
        """Returns non-fixed-point value cast to a specified type.
        
        Parameters
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
            Value represented in original format (not binary) casted according to `dtype`."""

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
                    val = utils.int_array(val)
                
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
        """Returns non-fixed-point value cast to a specified type.
        
        Parameters
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
            Value represented in original format (not binary) casted according to `dtype`."""

        if dtype is None:
            dtype = self.vdtype
        return self.astype(dtype, index, item)

    def raw(self):
        """Return raw signed fixed-point integer storage."""

        return self.val
    
    def uraw(self):
        """Return raw values encoded as unsigned two's-complement integers."""

        if np.iscomplexobj(self.val):
            raw_real = np.where(self.val.real < 0, (1 << self.n_word) + self.val.real, self.val.real)
            raw_imag = np.where(self.val.imag < 0, (1 << self.n_word) + self.val.imag, self.val.imag)
            return raw_real + 1j * raw_imag

        return np.where(self.val < 0, (1 << self.n_word) + self.val, self.val)

    def equal(self, x, index=None):
        """Sets the value of the Fxp using the value of other Fxp object.
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
            Fxp with it's value modified."""
        
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
        """Apply configured overflow behavior (saturate or wrap) to candidate raw values.
        
        Parameters
        ---
        new_val : numpy.ndarray
            Candidate raw values before overflow handling.
        val_min : int or float
            Minimum representable bound used by clipping/saturation logic.
        val_max : int or float
            Maximum representable bound used by clipping/saturation logic.
        
        Returns
        ---
        numpy.ndarray
            Raw values after applying configured overflow handling."""
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
        """Apply the configured rounding rule to intermediate numeric values.
        
        Parameters
        ---
        val : scalar, complex, array_like, or Fxp
            Value(s) used to initialize, convert, or assign fixed-point data.
        method : str, optional
            Computation method, typically `raw` or `repr` depending on configuration.
        
        Returns
        ---
        numpy.ndarray
            Rounded intermediate values using the configured rounding rule."""
        # NOTE ABOUT PASSTHROUGH FOR int/object INPUTS:
        # _round receives values in the raw-domain scale (val * conv_factor).
        # For >64-bit paths, set_val promotes arrays to dtype=object to preserve exact
        # integer precision, and those values are expected to already represent integer
        # raw quanta. For that reason we intentionally keep legacy passthrough behavior
        # for integer/object dtypes and skip fractional rounding logic in this branch.
        # If a non-integer value reaches this branch, conversion to stored raw integers
        # still occurs downstream in set_val via utils.int_array, preserving the
        # existing execution flow.
        val_dtype = getattr(val, 'dtype', None)
        if val_dtype is None and isinstance(val, np.generic):
            val_dtype = val.dtype

        is_integer = isinstance(val, int) or (val_dtype is not None and np.issubdtype(val_dtype, np.integer))
        is_object_dtype = val_dtype is not None and np.issubdtype(val_dtype, np.object_)

        if is_object_dtype and self.n_word < _n_word_max:
            import warnings as _warnings
            _warnings.warn('Object dtype reached rounding while n_word ({}) is smaller than n_word_max ({}). This usually indicates an unexpected object-typed input path and may bypass fractional rounding behavior.'.format(self.n_word, _n_word_max), RuntimeWarning, stacklevel=2)

        if is_integer or is_object_dtype:
            rval = val
        elif method == 'nearest_posinf' or method == 'nearest_neginf' or method == 'nearest_zero' or method == 'nearest_away':
            f = np.floor(val)
            frac = val - f
            gt = frac > 0.5
            eq = frac == 0.5
            if method == 'nearest_posinf':
                inc = gt | eq
            elif method == 'nearest_neginf':
                inc = gt
            elif method == 'nearest_zero':
                inc = gt | (eq & (val < 0))
            else:
                inc = gt | (eq & (val >= 0))
            rval = f + inc.astype(f.dtype)
        elif method == 'bit_trunc':
            rval = np.floor(val)
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
        """Execute registered callbacks for a named event and pass optional payload data.
        
        Parameters
        ---
        method : str
            Computation method, typically `raw` or `repr` depending on configuration.
        
        Side Effects
        ---
        Invokes registered callbacks for the specified event; callback implementations may perform external side effects."""
        if self.callbacks:
            for cb in self.callbacks:
                if hasattr(cb, method): getattr(cb, method)(self)

    # overloadings

    def __call__(self, val=None):
        """Set a new value and return the same `Fxp` instance.
        
        Parameters
        ---
        val : scalar, complex, array_like, or Fxp, optional
            Value(s) used to initialize, convert, or assign fixed-point data.
        
        Returns
        ---
        Fxp
            This instance, enabling chained assignment via call syntax."""
        if val is None:
            rval = self.get_val()
        else:
            rval = self.set_val(val)
        return rval

    def __len__(self):
        """Return the number of stored scalar elements."""
        return len(self.val)

    def __bool__(self):
        """Return `True` when at least one stored value is non-zero."""
        if self.size > 1:
            raise ValueError("The boolean value cannot be determined. Use any() or all().")
        else:
            return bool(self.get_val())

    def __int__(self):
        """Return the represented value cast to Python `int`."""
        if self.size > 1:
            raise TypeError('only length-1 arrays can be converted to Python scalars')
        return int(self.astype(int))

    def __float__(self):
        """Return the represented value cast to Python `float`."""
        if self.size > 1:
            raise TypeError('only length-1 arrays can be converted to Python scalars')
        return float(self.astype(float))

    def __complex__(self):
        """Return the represented value cast to Python `complex`."""
        if self.size > 1:
            raise TypeError('only length-1 arrays can be converted to Python scalars')
        return complex(self.astype(complex))
    
    # representation
    
    def __repr__(self):
        """Return a debug representation that includes dtype and value."""
        dtype_str = self.dtype
        data_str = str(self.get_val()).replace('\n', '\n '+' '*(len(dtype_str)))

        return '{}({})'.format(self.dtype, data_str)

    def __str__(self):
        """Return a user-facing string representation of the value."""
        return str(self.get_val())

    # numpy array representation - numpy hooks
    
    def __array__(self, *args, **kwargs):
        """Convert the represented value to a NumPy array.
        
        Parameters
        ---
        *args : tuple
            Positional arguments passed to wrapped NumPy handlers.
        **kwargs : dict
            Extra keyword arguments propagated to lower-level conversion or NumPy handlers.
        
        Returns
        ---
        numpy.ndarray
            NumPy array view/copy of represented values."""
        if self.config.array_op_method == 'raw':
            return np.asarray(self.val, *args, **kwargs)
        else:
            return np.asarray(self.get_val(), *args, **kwargs)
    
    def __array_wrap__(self, out_arr, context=None):
        """Wrap NumPy ufunc outputs back into `Fxp` when required.
        
        Parameters
        ---
        out_arr : numpy.ndarray
            Array produced by NumPy ufunc machinery before wrapping.
        context : tuple or None, optional
            NumPy array protocol context tuple.
        
        Returns
        ---
        Fxp or numpy.ndarray
            Wrapped output according to NumPy protocol and fxpmath output rules."""
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
        """Prepare NumPy ufunc outputs before computation.
        
        Parameters
        ---
        context : tuple or None, optional
            NumPy array protocol context tuple.
        *args : tuple
            Positional arguments passed to wrapped NumPy handlers.
        **kwargs : dict
            Extra keyword arguments propagated to lower-level conversion or NumPy handlers.
        
        Returns
        ---
        numpy.ndarray
            Prepared output array passed to NumPy ufunc execution."""
        if self.config.array_op_method == 'raw':
            return np.asarray(self.val, *args, **kwargs)
        else:
            return np.asarray(self.get_val(), *args, **kwargs)

    def __array_finalize__(self, obj):
        """Finalize metadata after NumPy creates array views.
        
        Parameters
        ---
        obj : numpy.ndarray or Fxp or None
            Source object provided by NumPy's array finalization hook.
        
        Side Effects
        ---
        Propagates dtype/config/status metadata when NumPy creates derived array views."""
        return
  
    # math operations
    
    def __neg__(self):
        """Return the arithmetic negation of the current value."""
        y = Fxp(-self.val, signed=self.signed, n_word=self.n_word, n_frac=self.n_frac, raw=True)
        return y

    def __pos__(self):
        """Return the unary plus result for the current value."""
        y = Fxp(+self.val, signed=self.signed, n_word=self.n_word, n_frac=self.n_frac, raw=True)
        return y

    def __abs__(self):
        """Return the absolute value preserving fixed-point behavior."""
        y = Fxp(abs(self.val), signed=self.signed, n_word=self.n_word, n_frac=self.n_frac, raw=True)
        return y          

    def __add__(self, x):
        """Compute element-wise addition with fixed-point sizing rules.
        
        Parameters
        ---
        x : Fxp, scalar, complex, or array_like
            Operand used in arithmetic, comparison, or dot-product operations.
        
        Returns
        ---
        Fxp
            New fixed-point object containing the operation result."""
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
        """Compute element-wise subtraction with fixed-point sizing rules.
        
        Parameters
        ---
        x : Fxp, scalar, complex, or array_like
            Operand used in arithmetic, comparison, or dot-product operations.
        
        Returns
        ---
        Fxp
            New fixed-point object containing the operation result."""
        from .functions import sub

        if not isinstance(x, Fxp):
            x = self._convert_op_input_value(x)
            _sizing = self.config.const_op_sizing
        else:
            _sizing = self.config.op_sizing

        return sub(self, x, out=self.config.op_out, out_like=self.config.op_out_like, sizing=_sizing, method=self.config.op_method)

    def __rsub__(self, x):
        """Compute reflected subtraction when `Fxp` is on the right-hand side.
        
        Parameters
        ---
        x : Fxp, scalar, complex, or array_like
            Operand used in arithmetic, comparison, or dot-product operations.
        
        Returns
        ---
        Fxp
            New fixed-point object containing the operation result."""
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
        """Compute element-wise multiplication with fixed-point sizing rules.
        
        Parameters
        ---
        x : Fxp, scalar, complex, or array_like
            Operand used in arithmetic, comparison, or dot-product operations.
        
        Returns
        ---
        Fxp
            New fixed-point object containing the operation result."""
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
        """Compute element-wise true division with fixed-point sizing rules.
        
        Parameters
        ---
        x : Fxp, scalar, complex, or array_like
            Operand used in arithmetic, comparison, or dot-product operations.
        
        Returns
        ---
        Fxp
            New fixed-point object containing the operation result."""
        from .functions import truediv

        if not isinstance(x, Fxp):
            x = self._convert_op_input_value(x)
            _sizing = self.config.const_op_sizing
        else:
            _sizing = self.config.op_sizing

        return truediv(self, x, out=self.config.op_out, out_like=self.config.op_out_like, sizing=_sizing, method=self.config.op_method)

    def __rtruediv__(self, x):
        """Compute reflected true division when `Fxp` is on the right-hand side.
        
        Parameters
        ---
        x : Fxp, scalar, complex, or array_like
            Operand used in arithmetic, comparison, or dot-product operations.
        
        Returns
        ---
        Fxp
            New fixed-point object containing the operation result."""
        from .functions import truediv

        if not isinstance(x, Fxp):
            x = self._convert_op_input_value(x)
            _sizing = self.config.const_op_sizing
        else:
            _sizing = self.config.op_sizing

        return truediv(x, self, out=self.config.op_out, out_like=self.config.op_out_like, sizing=_sizing, method=self.config.op_method)

    __itruediv__ = __truediv__

    def __floordiv__(self, x):
        """Compute element-wise floor division with fixed-point sizing rules.
        
        Parameters
        ---
        x : Fxp, scalar, complex, or array_like
            Operand used in arithmetic, comparison, or dot-product operations.
        
        Returns
        ---
        Fxp
            New fixed-point object containing the operation result."""
        from .functions import floordiv

        if not isinstance(x, Fxp):
            x = self._convert_op_input_value(x)
            _sizing = self.config.const_op_sizing
        else:
            _sizing = self.config.op_sizing

        return floordiv(self, x, out=self.config.op_out, out_like=self.config.op_out_like, sizing=_sizing, method=self.config.op_method)

    def __rfloordiv__(self, x):
        """Compute reflected floor division when `Fxp` is on the right-hand side.
        
        Parameters
        ---
        x : Fxp, scalar, complex, or array_like
            Operand used in arithmetic, comparison, or dot-product operations.
        
        Returns
        ---
        Fxp
            New fixed-point object containing the operation result."""
        from .functions import floordiv

        if not isinstance(x, Fxp):
            x = self._convert_op_input_value(x)
            _sizing = self.config.const_op_sizing
        else:
            _sizing = self.config.op_sizing

        return floordiv(x, self, out=self.config.op_out, out_like=self.config.op_out_like, sizing=_sizing, method=self.config.op_method)

    __ifloordiv__ = __floordiv__

    def __mod__(self, x):
        """Compute element-wise modulo with fixed-point sizing rules.
        
        Parameters
        ---
        x : Fxp, scalar, complex, or array_like
            Operand used in arithmetic, comparison, or dot-product operations.
        
        Returns
        ---
        Fxp
            New fixed-point object containing the operation result."""
        from .functions import mod

        if not isinstance(x, Fxp):
            x = self._convert_op_input_value(x)
            _sizing = self.config.const_op_sizing
        else:
            _sizing = self.config.op_sizing

        return mod(self, x, out=self.config.op_out, out_like=self.config.op_out_like, sizing=_sizing, method=self.config.op_method)

    def __rmod__(self, x):
        """Compute reflected modulo when `Fxp` is on the right-hand side.
        
        Parameters
        ---
        x : Fxp, scalar, complex, or array_like
            Operand used in arithmetic, comparison, or dot-product operations.
        
        Returns
        ---
        Fxp
            New fixed-point object containing the operation result."""
        from .functions import mod

        if not isinstance(x, Fxp):
            x = self._convert_op_input_value(x)
            _sizing = self.config.const_op_sizing
        else:
            _sizing = self.config.op_sizing

        return mod(x, self, out=self.config.op_out, out_like=self.config.op_out_like, sizing=_sizing, method=self.config.op_method)

    __imod__ = __mod__

    def __pow__(self, x):
        """Compute element-wise exponentiation with fixed-point sizing rules.
        
        Parameters
        ---
        x : Fxp, scalar, complex, or array_like
            Operand used in arithmetic, comparison, or dot-product operations.
        
        Returns
        ---
        Fxp
            New fixed-point object containing the operation result."""
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
        """Shift represented raw values to the right by the requested amount.
        
        Parameters
        ---
        n : int
            Bit-shift amount.
        
        Returns
        ---
        Fxp
            New fixed-point object containing the operation result."""
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
        """Shift represented raw values to the left by the requested amount.
        
        Parameters
        ---
        n : int
            Bit-shift amount.
        
        Returns
        ---
        Fxp
            New fixed-point object containing the operation result."""
        if self.config.shifting == 'expand':
            n_word = max(self.n_word, int(np.max(np.ceil(np.log2(np.abs(self.val)+0.5)))) + self.signed + n)
            new_value = self.val << np.array(n, dtype=self.val.dtype)
        elif self.config.shifting == 'keep':
            n_word = self.n_word

            @utils.array_support
            def _raw_lshift(val, nshift, nbits, signed):
                # Shift and keep only the least-significant nbits after shifting.
                val = (val << np.array(nshift, dtype=val.dtype)) % (1 << nbits)

                # If signed and MSb is 1, convert from two's complement representation.
                if signed and (int(val) & (1 << (nbits - 1))) != 0:
                    val = val - (1 << nbits)
                return val

            new_value = _raw_lshift(self.val, nshift=n, nbits=n_word, signed=self.signed)
        else:
            n_word = self.n_word
            new_value = self.val << np.array(n, dtype=self.val.dtype)

        y = Fxp(None, signed=self.signed, n_word=n_word, n_frac=self.n_frac)
        y.set_val(new_value, raw=True, vdtype=self.vdtype)   # set raw val shifted
        return y
    
    __ilshift__ = __lshift__

    def _bitwise_prepare_operand(self, x):
        """Validate and normalize right-side operand for bitwise operations."""
        if isinstance(x, Fxp):
            if self.n_word != x.n_word:
                raise ValueError("Operands dont't have same word size!")
            return x.val

        return x

    def _bitwise_binary_op(self, x, op):
        """Apply a binary bitwise helper with mixed/complex operand support."""
        x_val = self._bitwise_prepare_operand(x)
        out_val, force_complex_result = utils.binary_op_componentwise(
            self.val,
            x_val,
            op=op,
            n_word=self.n_word,
            warn_mixed=True,
            warning_stacklevel=5,
        )

        if self.signed:
            out_val = utils.twos_complement_componentwise(out_val, nbits=self.n_word)

        y = self.deepcopy()
        y.set_val(
            out_val,
            raw=True,
            vdtype=utils.bitwise_result_dtype(self.vdtype, force_complex=force_complex_result),
        )
        return y

    def __invert__(self):
        # inverted_val = ~ self.val

        """Apply bitwise inversion over represented raw values."""
        inverted_val, force_complex_result = utils.binary_invert_componentwise(self.val, n_word=self.n_word)

        if self.signed:
            inverted_val = utils.twos_complement_componentwise(inverted_val, nbits=self.n_word)

        y = self.deepcopy()
        y.set_val(
            inverted_val,
            raw=True,
            vdtype=utils.bitwise_result_dtype(self.vdtype, force_complex=force_complex_result),
        )
        return y

    def __and__(self, x):
        """Apply bitwise AND over represented raw values.
        
        Parameters
        ---
        x : Fxp, scalar, complex, or array_like
            Operand used in arithmetic, comparison, or dot-product operations.
        
        Returns
        ---
        Fxp
            New fixed-point object containing the operation result."""
        return self._bitwise_binary_op(x, utils.binary_and)

    __rand__ = __and__

    __iand__ = __and__

    def __or__(self, x):
        """Apply bitwise OR over represented raw values.
        
        Parameters
        ---
        x : Fxp, scalar, complex, or array_like
            Operand used in arithmetic, comparison, or dot-product operations.
        
        Returns
        ---
        Fxp
            New fixed-point object containing the operation result."""
        return self._bitwise_binary_op(x, utils.binary_or)

    __ror__ = __or__

    __ior__ = __or__

    def __xor__(self, x):
        """Apply bitwise XOR over represented raw values.
        
        Parameters
        ---
        x : Fxp, scalar, complex, or array_like
            Operand used in arithmetic, comparison, or dot-product operations.
        
        Returns
        ---
        Fxp
            New fixed-point object containing the operation result."""
        return self._bitwise_binary_op(x, utils.binary_xor)

    __rxor__ = __xor__

    __ixor__ = __xor__


    # comparisons

    def __lt__(self, x):
        """Compare whether values are strictly less than the operand.
        
        Parameters
        ---
        x : Fxp, scalar, complex, or array_like
            Operand used in arithmetic, comparison, or dot-product operations.
        
        Returns
        ---
        bool or numpy.ndarray
            Comparison result between represented values."""
        if isinstance(x, Fxp):
            x = x.get_val()
        return self.get_val() < x

    def __le__(self, x):
        """Compare whether values are less than or equal to the operand.
        
        Parameters
        ---
        x : Fxp, scalar, complex, or array_like
            Operand used in arithmetic, comparison, or dot-product operations.
        
        Returns
        ---
        bool or numpy.ndarray
            Comparison result between represented values."""
        if isinstance(x, Fxp):
            x = x.get_val()
        return self.get_val() <= x

    def __eq__(self, x):
        """Compare whether values are equal to the operand.
        
        Parameters
        ---
        x : Fxp, scalar, complex, or array_like
            Operand used in arithmetic, comparison, or dot-product operations.
        
        Returns
        ---
        bool or numpy.ndarray
            Comparison result between represented values."""
        if isinstance(x, Fxp):
            x = x.get_val()
        return self.get_val() == x

    def __ne__(self, x):
        """Compare whether values differ from the operand.
        
        Parameters
        ---
        x : Fxp, scalar, complex, or array_like
            Operand used in arithmetic, comparison, or dot-product operations.
        
        Returns
        ---
        bool or numpy.ndarray
            Comparison result between represented values."""
        if isinstance(x, Fxp):
            x = x.get_val()
        return self.get_val() != x

    def __gt__(self, x):
        """Compare whether values are strictly greater than the operand.
        
        Parameters
        ---
        x : Fxp, scalar, complex, or array_like
            Operand used in arithmetic, comparison, or dot-product operations.
        
        Returns
        ---
        bool or numpy.ndarray
            Comparison result between represented values."""
        if isinstance(x, Fxp):
            x = x.get_val()
        return self.get_val() > x

    def __ge__(self, x):
        """Compare whether values are greater than or equal to the operand.
        
        Parameters
        ---
        x : Fxp, scalar, complex, or array_like
            Operand used in arithmetic, comparison, or dot-product operations.
        
        Returns
        ---
        bool or numpy.ndarray
            Comparison result between represented values."""
        if isinstance(x, Fxp):
            x = x.get_val()
        return self.get_val() >= x

    # indexation
    def __getitem__(self, index):
        # return Fxp(self.val[index], like=self, raw=True)
        """Return one or more stored values selected by index.
        
        Parameters
        ---
        index : int, slice, tuple, or array_like
            Index selection used for element access/assignment.
        
        Returns
        ---
        Fxp
            Indexed fixed-point view/copy preserving dtype and configuration."""
        y = Fxp(like=self)
        y.val = self.val[index]
        return y

    def __setitem__(self, index, value):
        """Assign one or more stored values selected by index.
        
        Parameters
        ---
        index : int, slice, tuple, or array_like
            Index selection used for element access/assignment.
        value : scalar, complex, array_like, or Fxp
            Value(s) assigned into the indexed location.
        
        Side Effects
        ---
        Assigns selected elements in the underlying value storage through fixed-point quantization rules."""
        self.set_val(value, index=index)

    # get info about me
    def get_status(self, format=dict):
        """Return a shallow copy of current status flags (overflow, underflow, and inaccuracy).
        
        Parameters
        ---
        format : {dict, str}, optional
            Output format used for status reporting.
        
        Returns
        ---
        dict or str
            Status flags as dictionary or formatted string."""
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
        """Print a human-readable report of value, format, limits, and status information.
        
        Parameters
        ---
        verbose : int, optional
            Verbosity level for printed diagnostic information.
        
        Side Effects
        ---
        Prints a formatted diagnostic report to standard output."""
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
        """Return the current value formatted as a binary fixed-point representation.
        
        Parameters
        ---
        frac_dot : bool, optional
            Whether textual output includes the fractional-point separator.
        prefix : str, bool, or None, optional
            Prefix control for textual representations (`0b`/`0x`, custom, or disabled).
        
        Returns
        ---
        str or numpy.ndarray
            Binary fixed-point representation of stored values."""
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
        """Return the current value formatted as a hexadecimal fixed-point representation.
        
        Parameters
        ---
        padding : int, optional
            Minimum digit padding applied to textual representations.
        prefix : str, bool, or None, optional
            Prefix control for textual representations (`0b`/`0x`, custom, or disabled).
        
        Returns
        ---
        str or numpy.ndarray
            Hexadecimal fixed-point representation of stored values."""
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
        """Return a base-N representation string.
        
        Parameters
        ---
        base : int
            Numeric base used for textual conversion.
        frac_dot : bool, optional
            Whether textual output includes the fractional-point separator.
        
        Returns
        ---
        str or numpy.ndarray
            Base-N textual representation of stored values."""
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
        """Create an `Fxp` object from a binary representation string.
        
        Parameters
        ---
        val : scalar, complex, array_like, or Fxp
            Value(s) used to initialize, convert, or assign fixed-point data.
        raw : bool, optional
            When `True`, treat input values as raw stored integers instead of represented values.
        
        Returns
        ---
        Fxp
            This instance after parsing and assigning binary-formatted input."""
        self.set_val(utils.add_binary_prefix(val), raw=raw)
        return self

    # copy
    def copy(self):
        """Create a shallow copy of the object preserving configuration and stored values."""
        return copy.copy(self)

    def deepcopy(self):
        """Create a deep copy of the object preserving configuration and stored values."""
        return copy.deepcopy(self)

    def like(self, x):
        """Create a new `Fxp` object using this instance as template and optional new value.
        
        Parameters
        ---
        x : Fxp, scalar, complex, or array_like
            Operand used in arithmetic, comparison, or dot-product operations.
        
        Returns
        ---
        Fxp
            New fixed-point object with this instance as format/config template."""
        if isinstance(x, self.__class__):
            new_raw_val = self.val * 2**(x.n_frac - self.n_frac)
            return  x.copy().set_val(new_raw_val, raw=True)
        else:
            raise ValueError('`x` should be a Fxp object!')

    # reset
    def reset(self):
        #status (overwrite)
        """Reset status flags and clear overflow/underflow/inaccuracy indicators."""
        self.status = {
            'overflow': False,
            'underflow': False,
            'inaccuracy': False}

    def _convert_op_input_value(self, x, op_input_size=None):
        """Normalize operation operands into a compatible `Fxp` instance or scalar.
        
        Parameters
        ---
        x : Fxp, scalar, complex, or array_like
            Operand used in arithmetic, comparison, or dot-product operations.
        op_input_size : {'same', 'best'}, optional
            Sizing policy used when converting non-`Fxp` operands.
        
        Returns
        ---
        Fxp
            Operand converted to an `Fxp` compatible with operation sizing rules."""
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
        """Dispatch NumPy ufunc calls so `Fxp` participates in NumPy arithmetic protocols.
        
        Parameters
        ---
        ufunc : numpy.ufunc
            NumPy universal function being dispatched.
        method : str
            Computation method, typically `raw` or `repr` depending on configuration.
        *inputs : tuple
            Positional arguments passed to wrapped NumPy handlers.
        **kwargs : dict
            Extra keyword arguments propagated to lower-level conversion or NumPy handlers.
        
        Returns
        ---
        Fxp or numpy.ndarray or tuple or NotImplemented
            Dispatched ufunc result in the configured output form."""
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
        """Dispatch NumPy high-level function calls registered for `Fxp` handling.
        
        Parameters
        ---
        func : Callable
            NumPy function being dispatched through `__array_function__`.
        types : tuple
            Type signature tuple provided by NumPy `__array_function__` protocol.
        args : tuple
            Positional arguments passed to wrapped NumPy handlers.
        kwargs : dict
            Extra keyword arguments propagated to lower-level conversion or NumPy handlers.
        
        Returns
        ---
        object
            Result returned by the registered fxpmath NumPy-function implementation."""
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
        """Execute wrapped NumPy-compatible functions and normalize output typing.
        
        Parameters
        ---
        func : Callable
            NumPy function being dispatched through `__array_function__`.
        *args : tuple
            Positional arguments passed to wrapped NumPy handlers.
        **kwargs : dict
            Extra keyword arguments propagated to lower-level conversion or NumPy handlers.
        
        Returns
        ---
        object
            Normalized result of a wrapped NumPy-compatible function."""
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
        """Apply configured output typing rules for array-oriented operations.
        
        Parameters
        ---
        out_arr : numpy.ndarray
            Array produced by NumPy ufunc machinery before wrapping.
        
        Returns
        ---
        Fxp or numpy.ndarray
            Result converted to configured array output type."""
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
        """Test whether all array elements along a given axis evaluate to True.
        
                NumPy-compatible wrapper; output formatting follows the active `Config`.
        
        Parameters
        ---
        axis : int or tuple[int, ...], optional
            Axis or axes along which to apply the operation.
        **kwargs : dict
            Extra keyword arguments propagated to lower-level conversion or NumPy handlers.
        
        Returns
        ---
        bool or numpy.ndarray
            Logical AND reduction result."""
        return np.all(np.array(self), axis=axis, **kwargs)

    def any(self, axis=None, **kwargs):
        """Test whether any array element along a given axis evaluates to True.
        
                NumPy-compatible wrapper; output formatting follows the active `Config`.
        
        Parameters
        ---
        axis : int or tuple[int, ...], optional
            Axis or axes along which to apply the operation.
        **kwargs : dict
            Extra keyword arguments propagated to lower-level conversion or NumPy handlers.
        
        Returns
        ---
        bool or numpy.ndarray
            Logical OR reduction result."""
        return np.any(np.array(self), axis=axis, **kwargs)

    def argmax(self, axis=None, **kwargs):
        """Return indices of maximum values along an axis.
        
                NumPy-compatible wrapper; output formatting follows the active `Config`.
        
        Parameters
        ---
        axis : int or tuple[int, ...], optional
            Axis or axes along which to apply the operation.
        **kwargs : dict
            Extra keyword arguments propagated to lower-level conversion or NumPy handlers.
        
        Returns
        ---
        int or numpy.ndarray
            Indices of maximum values along the requested axis."""
        return np.argmax(self.val, axis=axis, **kwargs)    # operates over raw values

    def argmin(self, axis=None, **kwargs):
        """Return indices of minimum values along an axis.
        
                NumPy-compatible wrapper; output formatting follows the active `Config`.
        
        Parameters
        ---
        axis : int or tuple[int, ...], optional
            Axis or axes along which to apply the operation.
        **kwargs : dict
            Extra keyword arguments propagated to lower-level conversion or NumPy handlers.
        
        Returns
        ---
        int or numpy.ndarray
            Indices of minimum values along the requested axis."""
        return np.argmin(self.val, axis=axis, **kwargs)    # operates over raw values

    def argpartition(self, axis=-1, **kwargs):
        """Perform an indirect partition along the given axis using the algorithm specified by the `kind` keyword. It returns an array of indices of the same shape as `a` that index data along the given axis in partitioned order.
        
                NumPy-compatible wrapper; output formatting follows the active `Config`.
        
        Parameters
        ---
        axis : int or tuple[int, ...], optional
            Axis or axes along which to apply the operation.
        **kwargs : dict
            Extra keyword arguments propagated to lower-level conversion or NumPy handlers.
        
        Returns
        ---
        numpy.ndarray
            Indices that partition the array around selected kth elements."""
        return np.argpartition(self.val, axis=axis, **kwargs)    # operates over raw values

    def argsort(self, axis=-1, **kwargs):
        """Perform an indirect sort along the given axis using the algorithm specified by the `kind` keyword. It returns an array of indices of the same shape as `a` that index data along the given axis in sorted order.
        
                NumPy-compatible wrapper; output formatting follows the active `Config`.
        
        Parameters
        ---
        axis : int or tuple[int, ...], optional
            Axis or axes along which to apply the operation.
        **kwargs : dict
            Extra keyword arguments propagated to lower-level conversion or NumPy handlers.
        
        Returns
        ---
        numpy.ndarray
            Indices that sort the array along the selected axis."""
        return np.argsort(self.val, axis=axis, **kwargs)    # operates over raw values

    def nonzero(self):
        """Return the indices of the elements that are non-zero.
        
        NumPy-compatible wrapper; output formatting follows the active `Config`."""
        from .functions import nonzero
        return nonzero(self)  

    def max(self, axis=None, **kwargs):
        """Return the maximum of an array or maximum along an axis.
        
                NumPy-compatible wrapper; output formatting follows the active `Config`.
        
        Parameters
        ---
        axis : int or tuple[int, ...], optional
            Axis or axes along which to apply the operation.
        **kwargs : dict
            Extra keyword arguments propagated to lower-level conversion or NumPy handlers.
        
        Returns
        ---
        Fxp or numpy.ndarray
            Maximum value result in configured output form."""
        from .functions import fxp_max

        out = kwargs.pop('out', self.config.op_out)
        out_like = kwargs.pop('out_like', self.config.op_out_like)
        sizing = kwargs.pop('sizing', self.config.op_sizing)
        method = kwargs.pop('method', self.config.op_method)

        return fxp_max(self, axis=axis, out=out, out_like=out_like, sizing=sizing, method=method, **kwargs)

    def min(self, axis=None, **kwargs):
        """Return the minimum of an array or minimum along an axis.
        
                NumPy-compatible wrapper; output formatting follows the active `Config`.
        
        Parameters
        ---
        axis : int or tuple[int, ...], optional
            Axis or axes along which to apply the operation.
        **kwargs : dict
            Extra keyword arguments propagated to lower-level conversion or NumPy handlers.
        
        Returns
        ---
        Fxp or numpy.ndarray
            Minimum value result in configured output form."""
        from .functions import fxp_min

        out = kwargs.pop('out', self.config.op_out)
        out_like = kwargs.pop('out_like', self.config.op_out_like)
        sizing = kwargs.pop('sizing', self.config.op_sizing)
        method = kwargs.pop('method', self.config.op_method)

        return fxp_min(self, axis=axis, out=out, out_like=out_like, sizing=sizing, method=method, **kwargs)

    def mean(self, axis=None, **kwargs):
        """Compute the arithmetic mean along the specified axis.
        
                NumPy-compatible wrapper; output formatting follows the active `Config`.
        
        Parameters
        ---
        axis : int or tuple[int, ...], optional
            Axis or axes along which to apply the operation.
        **kwargs : dict
            Extra keyword arguments propagated to lower-level conversion or NumPy handlers.
        
        Returns
        ---
        Fxp or numpy.ndarray
            Arithmetic mean in configured output form."""
        if not 'out' in kwargs and \
            self.config.array_output_type == 'fxp' and \
            self.config.array_op_out is None and self.config.array_op_out_like is None: 
            
            kwargs['out'] = Fxp(like=self)  # define Fxp output with same size by default

        return np.mean(self, axis=axis, **kwargs)

    def std(self, axis=None, **kwargs):
        """Compute the standard deviation along the specified axis.
        
                NumPy-compatible wrapper; output formatting follows the active `Config`.
        
        Parameters
        ---
        axis : int or tuple[int, ...], optional
            Axis or axes along which to apply the operation.
        **kwargs : dict
            Extra keyword arguments propagated to lower-level conversion or NumPy handlers.
        
        Returns
        ---
        Fxp or numpy.ndarray
            Standard deviation in configured output form."""
        if not 'out' in kwargs and \
            self.config.array_output_type == 'fxp' and \
            self.config.array_op_out is None and self.config.array_op_out_like is None: 
            
            kwargs['out'] = Fxp(like=self)  # define Fxp output with same size by default

        return np.std(self, axis=axis, **kwargs)

    def var(self, axis=None, **kwargs):
        """Compute the variance along the specified axis.
        
                NumPy-compatible wrapper; output formatting follows the active `Config`.
        
        Parameters
        ---
        axis : int or tuple[int, ...], optional
            Axis or axes along which to apply the operation.
        **kwargs : dict
            Extra keyword arguments propagated to lower-level conversion or NumPy handlers.
        
        Returns
        ---
        Fxp or numpy.ndarray
            Variance in configured output form."""
        if not 'out' in kwargs and \
            self.config.array_output_type == 'fxp' and \
            self.config.array_op_out is None and self.config.array_op_out_like is None: 
            
            kwargs['out'] = Fxp(like=self)  # define Fxp output with same size by default

        return np.var(self, axis=axis, **kwargs)

    def sum(self, axis=None, **kwargs):
        """Sum of array elements over a given axis.
        
                NumPy-compatible wrapper; output formatting follows the active `Config`.
        
        Parameters
        ---
        axis : int or tuple[int, ...], optional
            Axis or axes along which to apply the operation.
        **kwargs : dict
            Extra keyword arguments propagated to lower-level conversion or NumPy handlers.
        
        Returns
        ---
        Fxp or numpy.ndarray
            Summation result in configured output form."""
        from .functions import sum

        out = kwargs.pop('out', self.config.op_out)
        out_like = kwargs.pop('out_like', self.config.op_out_like)
        sizing = kwargs.pop('sizing', self.config.op_sizing)
        method = kwargs.pop('method', self.config.op_method)

        return sum(self, axis=axis, out=out, out_like=out_like, sizing=sizing, method=method, **kwargs)

    def cumsum(self, axis=None, **kwargs):
        """Return the cumulative sum of the elements along a given axis.
        
                NumPy-compatible wrapper; output formatting follows the active `Config`.
        
        Parameters
        ---
        axis : int or tuple[int, ...], optional
            Axis or axes along which to apply the operation.
        **kwargs : dict
            Extra keyword arguments propagated to lower-level conversion or NumPy handlers.
        
        Returns
        ---
        Fxp or numpy.ndarray
            Cumulative sum in configured output form."""
        from .functions import cumsum

        out = kwargs.pop('out', self.config.op_out)
        out_like = kwargs.pop('out_like', self.config.op_out_like)
        sizing = kwargs.pop('sizing', self.config.op_sizing)
        method = kwargs.pop('method', self.config.op_method)

        return cumsum(self, axis=axis, out=out, out_like=out_like, sizing=sizing, method=method, **kwargs)

    def cumprod(self, axis=None, **kwargs):
        """Return the cumulative product of elements along a given axis.
        
                NumPy-compatible wrapper; output formatting follows the active `Config`.
        
        Parameters
        ---
        axis : int or tuple[int, ...], optional
            Axis or axes along which to apply the operation.
        **kwargs : dict
            Extra keyword arguments propagated to lower-level conversion or NumPy handlers.
        
        Returns
        ---
        Fxp or numpy.ndarray
            Cumulative product in configured output form."""
        from .functions import cumprod

        out = kwargs.pop('out', self.config.op_out)
        out_like = kwargs.pop('out_like', self.config.op_out_like)
        sizing = kwargs.pop('sizing', self.config.op_sizing)
        method = kwargs.pop('method', self.config.op_method)

        return cumprod(self, axis=axis, out=out, out_like=out_like, sizing=sizing, method=method, **kwargs)

    ravel = flatten

    def tolist(self):
        """Return represented values as nested Python lists.
        
        NumPy-compatible wrapper; output formatting follows the active `Config`."""
        return self.get_val().tolist()

    def sort(self, axis=-1, **kwargs):
        """Return a sorted copy of an array.
        
                NumPy-compatible wrapper; output formatting follows the active `Config`.
        
        Parameters
        ---
        axis : int or tuple[int, ...], optional
            Axis or axes along which to apply the operation.
        **kwargs : dict
            Extra keyword arguments propagated to lower-level conversion or NumPy handlers.
        
        Side Effects
        ---
        Sorts `self.val` in place using the underlying NumPy array sort."""
        self.val.sort(axis=axis, **kwargs)

    def conjugate(self, **kwargs):
        """Return the complex conjugate, element-wise.
        
                NumPy-compatible wrapper; output formatting follows the active `Config`.
        
        Parameters
        ---
        **kwargs : dict
            Extra keyword arguments propagated to lower-level conversion or NumPy handlers.
        
        Returns
        ---
        Fxp or numpy.ndarray
            Complex conjugate result in configured output form."""
        from .functions import conjugate

        out = kwargs.pop('out', self.config.op_out)
        out_like = kwargs.pop('out_like', self.config.op_out_like)
        sizing = kwargs.pop('sizing', self.config.op_sizing)
        method = kwargs.pop('method', self.config.op_method)

        return conjugate(self, out=out, out_like=out_like, sizing=sizing, method=method, **kwargs)

    conj = conjugate

    @property
    def T(self):
        """Transposed view of the value array.
        
        NumPy-compatible wrapper; output formatting follows the active `Config`."""
        x = self.copy()
        x.val = x.val.T
        return x    
    
    def transpose(self, axes=None, **kwargs):
        """Return an array with axes transposed.
        
                NumPy-compatible wrapper; output formatting follows the active `Config`.
        
        Parameters
        ---
        axes : tuple[int, ...], optional
            Axis permutation for transpose operations.
        **kwargs : dict
            Extra keyword arguments propagated to lower-level conversion or NumPy handlers.
        
        Returns
        ---
        Fxp
            Transposed fixed-point object."""
        from .functions import transpose

        out = kwargs.pop('out', self.config.op_out)
        out_like = kwargs.pop('out_like', self.config.op_out_like)
        sizing = kwargs.pop('sizing', self.config.op_sizing)
        method = kwargs.pop('method', self.config.op_method)

        return transpose(self, axes=axes, out=out, out_like=out_like, sizing=sizing, method=method, **kwargs)

    def item(self, *args):
        """Copy an element of an array to a standard Python scalar and return it.
        
                NumPy-compatible wrapper; output formatting follows the active `Config`.
        
        Parameters
        ---
        *args : tuple
            Positional arguments passed to wrapped NumPy handlers.
        
        Returns
        ---
        scalar
            Single represented element converted to a Python scalar."""
        if len(args) > 1:
            items = tuple(args)
        else:
            items = args[0]
        return self.astype(item=items)

    def clip(self, a_min=None, a_max=None, **kwargs):
        """Clip (limit) the values in an array.
        
                NumPy-compatible wrapper; output formatting follows the active `Config`.
        
        Parameters
        ---
        a_min : scalar or None, optional
            Lower clipping bound.
        a_max : scalar or None, optional
            Upper clipping bound.
        **kwargs : dict
            Extra keyword arguments propagated to lower-level conversion or NumPy handlers.
        
        Returns
        ---
        Fxp or numpy.ndarray
            Values clipped between provided bounds."""
        from .functions import clip

        out = kwargs.pop('out', self.config.op_out)
        out_like = kwargs.pop('out_like', self.config.op_out_like)
        sizing = kwargs.pop('sizing', self.config.op_sizing)
        method = kwargs.pop('method', self.config.op_method)

        return clip(self, a_min=a_min, a_max=a_max, out=out, out_like=out_like, sizing=sizing, method=method, **kwargs)

    def diagonal(self, offset=0, axis1=0, axis2=1, **kwargs):
        """Return specified diagonals.
        
                NumPy-compatible wrapper; output formatting follows the active `Config`.
        
        Parameters
        ---
        offset : int, optional
            Diagonal offset from the main diagonal.
        axis1 : int, optional
            First axis used for diagonal extraction.
        axis2 : int, optional
            Second axis used for diagonal extraction.
        **kwargs : dict
            Extra keyword arguments propagated to lower-level conversion or NumPy handlers.
        
        Returns
        ---
        Fxp or numpy.ndarray
            Extracted diagonal values in configured output form."""
        from .functions import diagonal

        out = kwargs.pop('out', self.config.op_out)
        out_like = kwargs.pop('out_like', self.config.op_out_like)
        sizing = kwargs.pop('sizing', self.config.op_sizing)
        method = kwargs.pop('method', self.config.op_method)

        return diagonal(self, offset=offset, axis1=axis1, axis2=axis2, out=out, out_like=out_like, sizing=sizing, method=method, **kwargs)  

    def trace(self, offset=0, axis1=0, axis2=1, **kwargs):
        """Return the sum along diagonals of the array.
        
                NumPy-compatible wrapper; output formatting follows the active `Config`.
        
        Parameters
        ---
        offset : int, optional
            Diagonal offset from the main diagonal.
        axis1 : int, optional
            First axis used for diagonal extraction.
        axis2 : int, optional
            Second axis used for diagonal extraction.
        **kwargs : dict
            Extra keyword arguments propagated to lower-level conversion or NumPy handlers.
        
        Returns
        ---
        Fxp or numpy.ndarray
            Trace result in configured output form."""
        from .functions import trace

        out = kwargs.pop('out', self.config.op_out)
        out_like = kwargs.pop('out_like', self.config.op_out_like)
        sizing = kwargs.pop('sizing', self.config.op_sizing)
        method = kwargs.pop('method', self.config.op_method)

        return trace(self, offset=offset, axis1=axis1, axis2=axis2, out=out, out_like=out_like, sizing=sizing, method=method, **kwargs) 

    def prod(self, axis=None, **kwargs):
        """Return the product of array elements over a given axis.
        
                NumPy-compatible wrapper; output formatting follows the active `Config`.
        
        Parameters
        ---
        axis : int or tuple[int, ...], optional
            Axis or axes along which to apply the operation.
        **kwargs : dict
            Extra keyword arguments propagated to lower-level conversion or NumPy handlers.
        
        Returns
        ---
        Fxp or numpy.ndarray
            Product reduction result in configured output form."""
        from .functions import prod

        out = kwargs.pop('out', self.config.op_out)
        out_like = kwargs.pop('out_like', self.config.op_out_like)
        sizing = kwargs.pop('sizing', self.config.op_sizing)
        method = kwargs.pop('method', self.config.op_method)

        return prod(self, axis=axis, out=out, out_like=out_like, sizing=sizing, method=method, **kwargs) 

    def dot(self, x, **kwargs):
        """Compute the dot product of two arrays.
        
                NumPy-compatible wrapper; output formatting follows the active `Config`.
        
        Parameters
        ---
        x : Fxp, scalar, complex, or array_like
            Operand used in arithmetic, comparison, or dot-product operations.
        **kwargs : dict
            Extra keyword arguments propagated to lower-level conversion or NumPy handlers.
        
        Returns
        ---
        Fxp or numpy.ndarray
            Dot-product result in configured output form."""
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
    """Configuration container that defines how an `Fxp` object behaves during conversion, arithmetic, and NumPy interoperability.
    
    `Config` stores policy-style options rather than value sizes. It controls overflow/rounding behavior, how
    non-`Fxp` operands are converted, how operation outputs are sized and typed, and how textual notations are formatted.
    Each `Fxp` instance owns a `config` object, so policies can be tuned per-variable for modeling different fixed-point
    pipelines.
    
    Main option groups:
    - Value processing: `overflow`, `rounding`, `shifting`.
    - Scalar operations: `op_method`, `op_input_size`, `op_sizing`, `const_op_sizing`, `op_out`, `op_out_like`.
    - NumPy/array operations: `array_output_type`, `array_op_method`, `array_op_out`, `array_op_out_like`.
    - Formatting: `dtype_notation`, `bin_prefix`, `hex_prefix`.
    
    Example
    ---
    >>> x = Fxp(2.0, True, 16, 4)
    >>> x.config.op_input_size = 'best'
    >>> x.config.const_op_sizing = 'fit'
    >>> x.config.array_output_type = 'array'
    >>> y = x + 0.125"""
    template = None

    def __init__(self, **kwargs):
        # size limits
        """Initialize configuration fields for an `Fxp` instance.
        
        Keyword arguments can override any supported policy field (for example `overflow`, `op_method`,
        `op_sizing`, `const_op_sizing`, `array_output_type`, `dtype_notation`, or output templates).
        Unspecified fields keep the defaults documented in `docs/config.md`.
        
        Parameters
        ---
        **kwargs : dict
            Configuration overrides keyed by field name.
        
        Side Effects
        ---
        Initializes all configuration fields, optionally applying values from keyword arguments and template objects.
        
        Examples
        ---
        >>> cfg = Config(op_method='raw', op_sizing='optimal', overflow='saturate')
        >>> cfg.rounding = 'trunc'"""
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
        """Return the maximum error used when inferring bit-widths from values."""
        return self._max_error
    
    @max_error.setter
    def max_error(self, val):
        """Set `max_error` configuration option.
        
        Parameters
        ---
        val : float
            Positive maximum absolute error tolerated when inferring best-fit sizes.
        
        Side Effects
        ---
        Validates and stores the `max_error` configuration value."""
        if val > 0:
            self._max_error = val
        else:
            raise ValueError('max_error must be greater than 0!')

    # n_word_max
    @property
    def n_word_max(self):
        """Return the maximum supported bit-width for integer operations."""
        return self._n_word_max
    
    @n_word_max.setter
    def n_word_max(self, val):
        """Set `n_word_max` configuration option.
        
        Parameters
        ---
        val : int
            Maximum allowed word length used by sizing and intermediate arithmetic checks.
        
        Side Effects
        ---
        Validates and stores the `n_word_max` configuration value."""
        if isinstance(val, int) and val > 0:
            self._n_word_max = val
        else:
            raise ValueError('n_word_max must be int type greater than 0!')

    # overflow
    @property
    def _overflow_list(self):
        """Return valid values for `overflow`."""
        return ['saturate', 'wrap']

    @property
    def overflow(self):
        """Return the selected overflow handling mode."""
        return self._overflow
    
    @overflow.setter
    def overflow(self, val):
        """Set `overflow` configuration option.
        
        Parameters
        ---
        val : {'saturate', 'wrap'}
            Overflow policy: clamp to representable range (`saturate`) or wrap modulo word length (`wrap`).
        
        Side Effects
        ---
        Validates and stores overflow handling mode."""
        if isinstance(val, str) and val in self._overflow_list:
            self._overflow = val
        else:
            raise ValueError('overflow must be str type with following valid values: {}'.format(self._overflow_list))

    # rounding
    @property
    def _rounding_list(self):
        """Return canonical valid values for `rounding`."""
        return ['around', 'nearest_posinf', 'nearest_neginf', 'nearest_zero', 'nearest_away', 'bit_trunc', 'floor', 'ceil', 'fix', 'trunc']

    @property
    def _rounding_aliases(self):
        """Return supported aliases for `rounding`."""
        return {
            # IEEE 754 + canonical names
            'nearest_even': 'around',
            'roundTiesToEven': 'around',
            'up': 'ceil',
            'roundTowardPositive': 'ceil',
            'down': 'floor',
            'roundTowardNegative': 'floor',
            'to_zero': 'trunc',
            'roundTowardZero': 'trunc',
            # IEEE 1666 / SystemC names where semantics match existing modes
            'SC_RND_CONV': 'around',
            'SC_RND': 'nearest_posinf',
            'SC_TRN_ZERO': 'trunc',
            # Descriptive aliases for SC_RND semantics
            'nearest_ties_to_posinf': 'nearest_posinf',
            'roundTiesToPositive': 'nearest_posinf',
            # Additional nearest tie-breaking modes
            'SC_RND_MIN_INF': 'nearest_neginf',
            'SC_RND_ZERO': 'nearest_zero',
            'SC_RND_INF': 'nearest_away',
            'SC_TRN': 'bit_trunc',
            'nearest_ties_to_neginf': 'nearest_neginf',
            'nearest_ties_to_zero': 'nearest_zero',
            'nearest_ties_away': 'nearest_away',
            'roundTiesToAway': 'nearest_away',
            'bit_truncation': 'bit_trunc',
        }

    @property
    def rounding(self):
        """Return the selected canonical rounding mode."""
        return self._rounding
    
    @rounding.setter
    def rounding(self, val):
        """Set `rounding` configuration option.
        
        Parameters
        ---
        val : str
            Rounding policy applied when represented values must be quantized to raw integers.
            Canonical modes are {'around', 'nearest_posinf', 'nearest_neginf', 'nearest_zero', 'nearest_away', 'bit_trunc', 'floor', 'ceil', 'fix', 'trunc'}.
            Accepted aliases include IEEE 754, IEEE 1666/SystemC, and canonical shorthand names where
            semantics match existing fxpmath modes.
        
        Side Effects
        ---
        Validates and stores rounding mode."""
        if not isinstance(val, str):
            raise ValueError('rounding must be str type with following valid values: {} and aliases: {}'.format(
                self._rounding_list, sorted(self._rounding_aliases.keys())
            ))

        if val in self._rounding_list:
            self._rounding = val
            return

        if val in self._rounding_aliases:
            self._rounding = self._rounding_aliases[val]
            return

        val_lower = val.lower()
        lower_aliases = {k.lower(): v for k, v in self._rounding_aliases.items()}
        if val_lower in lower_aliases:
            self._rounding = lower_aliases[val_lower]
            return

        raise ValueError('rounding must be str type with following valid values: {} and aliases: {}'.format(
            self._rounding_list, sorted(self._rounding_aliases.keys())
        ))
    # shifting
    @property
    def _shifting_list(self):
        """Return valid values for `shifting`."""
        return ['expand', 'trunc', 'keep']

    @property
    def shifting(self):
        """Return the selected shifting mode."""
        return self._shifting
    
    @shifting.setter
    def shifting(self, val):
        """Set `shifting` configuration option.
        
        Parameters
        ---
        val : {'expand', 'trunc', 'keep'}
            Policy used for shift operations that would otherwise exceed current format limits.
        
        Side Effects
        ---
        Validates and stores shift behavior mode."""
        if isinstance(val, str) and val in self._shifting_list:
            self._shifting = val
        else:
            raise ValueError('shifting must be str type with following valid values: {}'.format(self._shifting_list))

    # op_input_size
    @property
    def _op_input_size_list(self):
        """Return valid values for `op_input_size`."""
        return ['same', 'best']

    @property
    def op_input_size(self):
        """Return the selected operation input-size policy."""
        return self._op_input_size
    
    @op_input_size.setter
    def op_input_size(self, val):
        """Set `op_input_size` configuration option.
        
        Parameters
        ---
        val : {'same', 'best'}
            Conversion policy for non-`Fxp` operands in arithmetic operations.
        
        Side Effects
        ---
        Validates and stores non-`Fxp` operand sizing policy."""
        if isinstance(val, str) and val in self._op_input_size_list:
            self._op_input_size = val
        else:
            raise ValueError('op_input_size must be str type with following valid values: {}'.format(self._op_input_size_list))
    

    # op_out
    @property
    def op_out(self):
        """Return the configured default output target for scalar operations."""
        return self._op_out
    
    @op_out.setter
    def op_out(self, val):
        """Set `op_out` configuration option.
        
        Parameters
        ---
        val : Fxp or None
            Default destination object for scalar operations. `None` disables default redirection.
        
        Side Effects
        ---
        Stores default scalar-operation output target."""
        if val is None or isinstance(val, Fxp):
            self._op_out = val
        else:
            raise ValueError('op_out must be a Fxp object or None!')

    # op_out_like
    @property
    def op_out_like(self):
        """Return the configured default output template for scalar operations."""
        return self._op_out_like
    
    @op_out_like.setter
    def op_out_like(self, val):
        """Set `op_out_like` configuration option.
        
        Parameters
        ---
        val : Fxp or None
            Default template object used to construct scalar-operation outputs.
        
        Side Effects
        ---
        Stores default scalar-operation output template."""
        if val is None or isinstance(val, Fxp):
            self._op_out_like = val
        else:
            raise ValueError('op_out_like must be a Fxp object or None!')

    # op_sizing
    @property
    def _op_sizing_list(self):
        """Return valid values for `op_sizing`."""
        return ['optimal', 'same', 'fit', 'largest', 'smallest']

    @property
    def op_sizing(self):
        """Return the selected operation output-sizing policy."""
        return self._op_sizing
    
    @op_sizing.setter
    def op_sizing(self, val):
        """Set `op_sizing` configuration option.
        
        Parameters
        ---
        val : {'optimal', 'same', 'fit', 'largest', 'smallest'}
            Sizing rule for outputs of operations between `Fxp` operands.
        
        Side Effects
        ---
        Validates and stores scalar-operation output sizing policy."""
        if isinstance(val, str) and val in self._op_sizing_list:
            self._op_sizing = val
        else:
            raise ValueError('op_sizing must be str type with following valid values: {}'.format(self._op_sizing_list))

    # op_method
    @property
    def _op_method_list(self):
        """Return valid values for `op_method`."""
        return ['raw', 'repr']

    @property
    def op_method(self):
        """Return the selected scalar operation compute method."""
        return self._op_method
    
    @op_method.setter
    def op_method(self, val):
        """Set `op_method` configuration option.
        
        Parameters
        ---
        val : {'raw', 'repr'}
            Arithmetic kernel: integer-domain fixed-point path (`raw`) or represented-value path (`repr`).
        
        Side Effects
        ---
        Validates and stores scalar-operation method."""
        if isinstance(val, str) and val in self._op_method_list:
            self._op_method = val
        else:
            raise ValueError('op_method must be str type with following valid values: {}'.format(self._op_method_list))

    # const_op_sizing
    @property
    def _const_op_sizing_list(self):
        """Return valid values for `const_op_sizing`."""
        return ['optimal', 'same', 'fit', 'largest', 'smallest']

    @property
    def const_op_sizing(self):
        """Return the selected constant-operation sizing policy."""
        return self._const_op_sizing
    
    @const_op_sizing.setter
    def const_op_sizing(self, val):
        """Set `const_op_sizing` configuration option.
        
        Parameters
        ---
        val : {'optimal', 'same', 'fit', 'largest', 'smallest'}
            Sizing rule for operations where one operand is a non-`Fxp` constant.
        
        Side Effects
        ---
        Validates and stores constant-operation sizing policy."""
        if isinstance(val, str) and val in self._const_op_sizing_list:
            self._const_op_sizing = val
        else:
            raise ValueError('op_sizing must be str type with following valid values: {}'.format(self._const_op_sizing_list))

    # array_output_type
    @property
    def _array_output_type_list(self):
        """Return valid values for `array_output_type`."""
        return ['fxp', 'array']

    @property
    def array_output_type(self):
        """Return the selected array-output typing policy."""
        return self._array_output_type
    
    @array_output_type.setter
    def array_output_type(self, val):
        """Set `array_output_type` configuration option.
        
        Parameters
        ---
        val : {'fxp', 'array'}
            Output container type returned by array-style NumPy operations.
        
        Side Effects
        ---
        Validates and stores array-output type policy."""
        if isinstance(val, str) and val in self._array_output_type_list:
            self._array_output_type = val
        else:
            raise ValueError('array_output_type must be str type with following valid values: {}'.format(self._array_output_type_list))

    # array_op_out
    @property
    def array_op_out(self):
        """Return the configured default output target for array operations."""
        return self._array_op_out
    
    @array_op_out.setter
    def array_op_out(self, val):
        """Set `array_op_out` configuration option.
        
        Parameters
        ---
        val : Fxp or None
            Default destination object for array operations. `None` disables default redirection.
        
        Side Effects
        ---
        Stores default array-operation output target."""
        if val is None or isinstance(val, Fxp):
            self._array_op_out = val
        else:
            raise ValueError('array_op_out must be a Fxp object or None!')

    # array_op_out_like
    @property
    def array_op_out_like(self):
        """Return the configured default output template for array operations."""
        return self._array_op_out_like
    
    @array_op_out_like.setter
    def array_op_out_like(self, val):
        """Set `array_op_out_like` configuration option.
        
        Parameters
        ---
        val : Fxp or None
            Default template object used to construct array-operation outputs.
        
        Side Effects
        ---
        Stores default array-operation output template."""
        if val is None or isinstance(val, Fxp):
            self._array_op_out_like = val
        else:
            raise ValueError('array_op_out_like must be a Fxp object or None!')

    # array_op_method
    @property
    def _array_op_method_list(self):
        """Return valid values for `array_op_method`."""
        return ['raw', 'repr']

    @property
    def array_op_method(self):
        """Return the selected array operation compute method."""
        return self._array_op_method
    
    @array_op_method.setter
    def array_op_method(self, val):
        """Set `array_op_method` configuration option.
        
        Parameters
        ---
        val : {'raw', 'repr'}
            Arithmetic kernel for array operations (`raw` for integer-domain operations, `repr` for represented values).
        
        Side Effects
        ---
        Validates and stores array-operation method."""
        if isinstance(val, str) and val in self._array_op_method_list:
            self._array_op_method = val
        else:
            raise ValueError('array_op_method must be str type with following valid values: {}'.format(self._array_op_method_list))

    # dtype_notation
    @property
    def _dtype_notation_list(self):
        """Return valid values for `dtype_notation`."""
        return ['fxp', 'Q']

    @property
    def dtype_notation(self):
        """Return the selected dtype notation style."""
        return self._dtype_notation
    
    @dtype_notation.setter
    def dtype_notation(self, val):
        """Set `dtype_notation` configuration option.
        
        Parameters
        ---
        val : {'fxp', 'Q'}
            Preferred notation used by dtype-formatting helpers.
        
        Side Effects
        ---
        Validates and stores dtype-notation style."""
        if isinstance(val, str) and val in self._dtype_notation_list:
            self._dtype_notation = val
        else:
            raise ValueError('dtype_notation must be str type with following valid values: {}'.format(self._dtype_notation_list))

    # prefixes
    @property
    def bin_prefix(self):
        """Return whether binary string outputs include the `0b` prefix."""
        return self._bin_prefix
    
    @bin_prefix.setter
    def bin_prefix(self, prefix):
        """Set `bin_prefix` configuration option.
        
        Parameters
        ---
        prefix : str or None
            Prefix string to prepend in formatted outputs, or `None` to disable prefixing.
        
        Side Effects
        ---
        Validates and stores binary-prefix formatting configuration."""
        if prefix is not None and not isinstance(prefix, str):
            print("Warning: the prefix should be a string, converted to string automatically!")
            prefix = str(prefix)

        if prefix not in [None, 'b', '0b', 'B', '0B']:
            print(f"Warning: the prefix {prefix} is not a common prefix for binary values!")

        self._bin_prefix = prefix

    @property
    def hex_prefix(self):
        """Return whether hexadecimal string outputs include the `0x` prefix."""
        return self._hex_prefix
    
    @hex_prefix.setter
    def hex_prefix(self, prefix):
        """Set `hex_prefix` configuration option.
        
        Parameters
        ---
        prefix : str or None
            Prefix string to prepend in formatted outputs, or `None` to disable prefixing.
        
        Side Effects
        ---
        Validates and stores hexadecimal-prefix formatting configuration."""
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
        """Print all configuration fields and their current values."""
        for k, v in self.__dict__.items():
            print('\t{: <24}:\t{}'.format(k.strip('_'), v))

    def update(self, **kwargs):
        """Update one or more configuration fields from keyword arguments.
        
        Parameters
        ---
        **kwargs : dict
            Extra keyword arguments propagated to lower-level conversion or NumPy handlers.
        
        Side Effects
        ---
        Updates one or more configuration fields from provided keyword arguments."""
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

    # copy
    def copy(self):
        """Create a shallow copy of the object preserving configuration and stored values."""
        return copy.copy(self)

    def deepcopy(self):
        """Create a deep copy of the object preserving configuration and stored values."""
        return copy.deepcopy(self)
    # endregion

# ----------------------------------------------------------------------------------------
# Internal functions
# ----------------------------------------------------------------------------------------
def implements(*np_functions):
   """Register `Fxp` handlers for one or more NumPy functions.
   
   Parameters
   ---
   *np_functions : Callable
       NumPy function objects used as dispatch keys in `_NUMPY_HANDLED_FUNCTIONS`.
   
   Returns
   ---
   Callable
       Decorator that binds an fxpmath implementation to each provided NumPy function."""
   def decorator(fxp_func):
        """Decorator used to bind a function to NumPy dispatch.
        
        Parameters
        ---
        fxp_func : Callable
            fxpmath implementation that will handle the registered NumPy function(s).
        
        Returns
        ---
        Callable
            The same function passed in `fxp_func`, after registration side effects are applied."""
        for np_func in np_functions:
            _NUMPY_HANDLED_FUNCTIONS[np_func] = fxp_func
        return fxp_func
   return decorator
