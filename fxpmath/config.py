"""Configuration container for Fxp behavior and operation policies."""

import copy


def _is_fxp_instance(val):
    """Late-import Fxp to avoid circular imports with fxpmath.objects."""
    from .objects import Fxp

    return _is_fxp_instance(val)

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
        if val is None or _is_fxp_instance(val):
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
        if val is None or _is_fxp_instance(val):
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
        if val is None or _is_fxp_instance(val):
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
        if val is None or _is_fxp_instance(val):
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
