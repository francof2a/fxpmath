import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import fxpmath as fxp
from fxpmath.utils import *

import numpy as np
import warnings

import fxpmath.utils.numeric as numeric_utils

def test_strbin2int():
    """Validates strbin2int by checking binary representation/interpretation paths."""
    assert strbin2int('0000') == 0
    assert strbin2int('0001') == 1
    assert strbin2int('1000', signed=False) == 8
    assert strbin2int('0b0010') == 2
    assert strbin2int('0b0100 1101') == 77
    assert strbin2int('0b1000 0000') == -128
    assert strbin2int('0b1111 1111') == -1
    assert strbin2int('0b1111 1110') == -2
    assert strbin2int('0b0111 1111') == 127
    assert strbin2int('0b1000 0000', signed=False) == 128
    assert strbin2int('0b0000', n_word=8) == 0
    assert strbin2int('0b0100', n_word=8) == 4
    assert strbin2int('0b1111', n_word=8) == -1
    assert strbin2int('0b1000', n_word=8) == strbin2int('0b1111 1000', n_word=8)

def test_strbin2float():
    """Validates strbin2float by checking binary representation/interpretation paths."""
    assert strbin2float('0001') == 1.0
    assert strbin2float('0b1000 0000') == -128.0
    assert strbin2float('0b0111 1111') == 127.0
    assert strbin2float('000.1') == 0.5
    assert strbin2float('b010.101') == 2.625
    assert strbin2float('111.1') == -0.5
    assert strbin2float('0001', n_frac=1) == 0.5
    assert strbin2float('0001', n_frac=2) == 0.25
    assert strbin2float('000.1', n_frac=4) == 0.5

def test_strbin2complex():
    """Validates strbin2complex by checking binary representation/interpretation paths, complex fixed-point behavior."""
    assert strbin2complex('0001') == 1.0 + 1j*0
    assert strbin2complex('0b10000000j') == -1j*128.0
    assert strbin2complex('0b01+0b10000000j') == 1.0 - 1j*128.0
    assert strbin2complex('0b1+0b1000 0000j', signed=False) == 1.0 + 1j*128.0
    assert strbin2complex('0b1 - 0b1000 0000j', signed=False) == 1.0 - 1j*128.0

def test_strhex2int():
    """Validates strhex2int by checking hexadecimal parsing/formatting paths."""
    assert strhex2int('0x00') == 0
    assert strhex2int('0x0A') == 10
    assert strhex2int('0x7F') == 127
    assert strhex2int('0xFF') == -1
    assert strhex2int('0x0F') == 15
    assert strhex2int('0x80') == -128
    assert strhex2int('0x80', signed=False) == 128
    assert strhex2int('0xFF', signed=False) == 255
    assert strhex2int('0x100') == 256

def test_strhex2float():
    """Validates strhex2float by checking hexadecimal parsing/formatting paths."""
    assert strhex2float('0x00') == 0.0
    assert strhex2float('0x0A') == 10.0
    assert strhex2float('0x7F') == 127.0
    assert strhex2float('0xFF') == -1.0
    assert strhex2float('0x0F') == 15.0
    assert strhex2float('0x80') == -128.0

    assert strhex2float('0x00', n_frac=2) == 0.0
    assert strhex2float('0x0A', n_frac=2) == 2.5
    assert strhex2float('0x7F', n_frac = 3) == 16.0 - 0.125
    assert strhex2float('0xFF', n_frac = 4) == 0 - 0.0625
    assert strhex2float('0x0F', n_frac = 4) == 1 - 0.0625
    assert strhex2float('0x80', n_frac = 4) == -8.0

def test_str2num():
    # int
    """Validates str2num by checking hexadecimal parsing/formatting paths, binary representation/interpretation paths, NumPy interoperability."""
    assert str2num('0') == 0
    assert str2num('10') == 10
    assert str2num('253') == 253
    assert str2num('-5') == -5
    # floats
    assert str2num('7.2') == 7.2
    assert str2num('-7.2') == -7.2
    # binary
    assert str2num('0b0110') == 6
    assert str2num('0b1110') == -2
    # hex
    assert str2num('0x7F') == 127
    assert str2num('0x80') == -128.0
    assert str2num('0x0A', n_frac=2) == 2.5
    # list
    assert str2num(['10', '-7.2', '0b0110', '0x7F']) == [10, -7.2, 6, 127]
    # transparent types
    assert (str2num(np.array([0,1,2,3])) == np.array([0,1,2,3])).all()
    assert str2num(None) is None

def test_binary_repr():
    """Verify binary formatter output across padding and fractional-point placement scenarios."""
    assert binary_repr(6, n_word=8) == '00000110'
    assert binary_repr(6, n_word=8, n_frac=3) == '00000.110'
    assert binary_repr(6, n_word=8, n_frac=0) == '00000110.'
    assert binary_repr(6, n_word=8, n_frac=-2) == '00000110##.'
    assert binary_repr(6, n_word=8, n_frac=8) == '.00000110'

    assert binary_repr(-1, n_word=4) == '1111'

def test_base_repr():
    """Verify generic base formatter for positive/negative integers and fractional-point formatting."""
    assert base_repr(6) == '110'
    assert base_repr(6, base=2, n_frac=3) == '.110'
    assert base_repr(6, n_frac=0) == '110.'
    assert base_repr(6, n_frac=-2) == '110##.'
    assert base_repr(6, n_frac=8) == '.00000110'

    assert base_repr(-1,) == '-1'
    assert base_repr(-6,) == '-110'
    assert base_repr(-6, n_frac=1) == '-11.0'

    assert base_repr(30, base=16) == '1E'
    assert base_repr(-30, base=16) == '-1E'

def test_bits_len():
    """Verify bit-length helper for positive and negative integers with optional signed-bit accounting."""
    assert bits_len(1) == 1
    assert bits_len(-1) == 1
    assert bits_len(1, signed=True) == 2
    assert bits_len(31) == 5
    assert bits_len(32) == 6
    assert bits_len(-32) == 6
    assert bits_len(-33) == 7

def test_add_binary_prefix():
    # single values
    """Validates add binary prefix by checking binary representation/interpretation paths, complex fixed-point behavior, NumPy interoperability."""
    assert add_binary_prefix('0') == '0b0'
    assert add_binary_prefix('1') == '0b1'
    assert add_binary_prefix('b0') == '0b0'
    assert add_binary_prefix('b1') == '0b1'
    assert add_binary_prefix('0b0') == '0b0'
    assert add_binary_prefix('0b1') == '0b1'

    assert add_binary_prefix('0110') == '0b0110'
    assert add_binary_prefix('b1111') == '0b1111'

    assert add_binary_prefix('01.001') == '0b01.001'
    assert add_binary_prefix('00.000') == '0b00.000'

    # list and arrays
    assert np.all(add_binary_prefix(['110', '001']) == np.array(['0b110', '0b001']))
    assert np.all(
        add_binary_prefix([['110', '001'], ['b111', '0b101']]) == \
            np.array([['0b110', '0b001'], ['0b111', '0b101']])
        )
    assert np.all(add_binary_prefix(np.array(['110', '001'])) == np.array(['0b110', '0b001']))
    assert np.all(
        add_binary_prefix(np.array([['110', '001'], ['b111', '0b101']])) == \
            np.array([['0b110', '0b001'], ['0b111', '0b101']])
        )
    
    # complex
    assert add_binary_prefix('0110+1110j') == '0b0110+0b1110j'
    assert add_binary_prefix('0110-1110j') == '0b0110-0b1110j'
    assert add_binary_prefix('0b0110-1110j') == '0b0110-0b1110j'
    assert add_binary_prefix('0110-0b1110j') == '0b0110-0b1110j'
    assert np.all(add_binary_prefix(['110+101j', '001-111j']) == np.array(['0b110+0b101j', '0b001-0b111j']))

    # test wrong input formats
    inputs_list = [0, 1, 3, '3', '102', '0b1102']
    for i in inputs_list:
        try:
            _ = add_binary_prefix(i)
        except:
            assert True
        else:
            print(f"input processed right when should be wrong: {i}")
            assert False

def test_complex_repr():
    """Validates complex repr by checking complex fixed-point behavior, NumPy interoperability."""
    assert complex_repr('2', '-4.5') == '2-4.5j'
    assert complex_repr('2', '4.5') == '2+4.5j'
    assert complex_repr('-2', '4.5') == '-2+4.5j'

    assert np.all(complex_repr(['1', '-2.5'], ['4.5', '0']) == np.array(['1+4.5j', '-2.5+0j']))
    


def test_clip_fast_path_where_preserves_unmasked_values():
    """Verify ndarray fast-path clipping keeps unmasked values when `where` is used."""
    x = np.array([1.0, 5.0, 3.0])
    y = numeric_utils.clip(x, val_min=2.0, val_max=4.0, where=np.array([True, False, True]))

    assert isinstance(y, np.ndarray)
    assert np.all(y == np.array([2.0, 5.0, 3.0]))


def test_clip_fallback_matches_vectorized_for_legacy_inputs():
    """Verify list, tuple, scalar, and object-array inputs keep legacy clip semantics."""
    cases = [
        [1, 9, -3],
        (1, 9, -3),
        2**70,
        np.array([2**70, -(2**70), 5], dtype=object),
    ]

    for x in cases:
        y = numeric_utils.clip(x, val_min=-2, val_max=4)
        y_ref = numeric_utils.clip_vectorized(x, val_min=-2, val_max=4)
        assert np.array_equal(np.asarray(y, dtype=object), np.asarray(y_ref, dtype=object))


def test_clip_fallback_warns_once_when_kwargs_are_ignored():
    """Verify unsupported kwargs are ignored once on the legacy fallback path with a one-time warning."""
    numeric_utils._CLIP_KWARGS_WARNING_SHOWN = False

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        y_list = numeric_utils.clip([1, 5, 3], val_min=2, val_max=4, where=[True, False, True])
        y_tuple = numeric_utils.clip((1, 5, 3), val_min=2, val_max=4, where=[True, False, True])
        y_object = numeric_utils.clip(np.array([1, 5, 3], dtype=object), val_min=2, val_max=4, where=[True, False, True])

    assert len(caught) == 1
    assert 'ignored keyword arguments' in str(caught[0].message)
    assert np.array_equal(np.asarray(y_list), np.asarray(numeric_utils.clip_vectorized([1, 5, 3], 2, 4)))
    assert np.array_equal(np.asarray(y_tuple), np.asarray(numeric_utils.clip_vectorized((1, 5, 3), 2, 4)))
    assert np.array_equal(
        np.asarray(y_object, dtype=object),
        np.asarray(numeric_utils.clip_vectorized(np.array([1, 5, 3], dtype=object), 2, 4), dtype=object),
    )
