import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fxpmath.objects import Fxp

import numpy as np
import warnings

def test_rounding_aliases():
    """Validates alias names for supported rounding modes across IEEE 754 / IEEE 1666 / canonical naming."""
    aliases_to_canonical = {
        'nearest_even': 'around',
        'roundTiesToEven': 'around',
        'SC_RND_CONV': 'around',
        'up': 'ceil',
        'roundTowardPositive': 'ceil',
        'down': 'floor',
        'roundTowardNegative': 'floor',
        'to_zero': 'trunc',
        'roundTowardZero': 'trunc',
        'SC_TRN_ZERO': 'trunc',
        'SC_RND': 'nearest_posinf',
        'nearest_ties_to_posinf': 'nearest_posinf',
        'roundTiesToPositive': 'nearest_posinf',
        'SC_RND_MIN_INF': 'nearest_neginf',
        'nearest_ties_to_neginf': 'nearest_neginf',
        'SC_RND_ZERO': 'nearest_zero',
        'nearest_ties_to_zero': 'nearest_zero',
        'SC_RND_INF': 'nearest_away',
        'nearest_ties_away': 'nearest_away',
        'roundTiesToAway': 'nearest_away',
        'SC_TRN': 'bit_trunc',
        'bit_truncation': 'bit_trunc',
    }

    tie_val = 1.375  # exactly halfway between two representable values when n_frac=2
    for alias, canonical in aliases_to_canonical.items():
        x_alias = Fxp(tie_val, True, 8, 2, rounding=alias)
        x_canon = Fxp(tie_val, True, 8, 2, rounding=canonical)
        assert x_alias.config.rounding == canonical
        assert x_alias() == x_canon()

    # Existing names remain valid and preserve current canonical storage.
    assert Fxp(1.1, True, 8, 2, rounding='fix').config.rounding == 'fix'
    assert Fxp(1.1, True, 8, 2, rounding='trunc').config.rounding == 'trunc'


def test_rounding_remaining_modes_comprehensive():
    """Validates remaining new modes across scalar/array/complex and >64-bit configurations."""

    def expected_real(values, n_frac, mode):
        scale = 2 ** n_frac
        vals = np.asarray(values)
        scaled = vals * scale
        if mode == 'nearest_neginf':
            q = np.ceil(scaled - 0.5)
        elif mode == 'nearest_zero':
            q = np.where(scaled >= 0, np.ceil(scaled - 0.5), np.floor(scaled + 0.5))
        elif mode == 'nearest_away':
            q = np.where(scaled >= 0, np.floor(scaled + 0.5), np.ceil(scaled - 0.5))
        elif mode == 'bit_trunc':
            q = np.floor(scaled)
        else:
            raise ValueError(mode)
        return q / scale

    def expected_complex(values, n_frac, mode):
        vals = np.asarray(values)
        real_q = expected_real(vals.real, n_frac, mode)
        imag_q = expected_real(vals.imag, n_frac, mode)
        return real_q + 1j * imag_q

    mode_to_aliases = {
        'nearest_neginf': ['SC_RND_MIN_INF', 'nearest_ties_to_neginf'],
        'nearest_zero': ['SC_RND_ZERO', 'nearest_ties_to_zero'],
        'nearest_away': ['SC_RND_INF', 'nearest_ties_away', 'roundTiesToAway'],
        'bit_trunc': ['SC_TRN', 'bit_truncation'],
    }

    configs = [
        dict(signed=True, n_word=8, n_frac=2),
        dict(signed=True, n_word=80, n_frac=70),   # >64-bit
        dict(signed=False, n_word=72, n_frac=64),  # >64-bit unsigned
    ]

    for mode, aliases in mode_to_aliases.items():
        for cfg in configs:
            step = 2.0 ** (-cfg['n_frac'])
            if cfg['signed']:
                values = np.array([0.0, 0.5 * step, -0.5 * step, 3.5 * step, -3.5 * step, 0.1, -0.1])
            else:
                values = np.array([0.0, 0.5 * step, 3.5 * step, 0.1, 0.2])

            for v in values:
                x = Fxp(float(v), rounding=mode, **cfg)
                expected = expected_real(float(v), cfg['n_frac'], mode)
                assert np.isclose(x(), expected)

            arr_2d = np.vstack([values, values + step]) if values.size > 1 else np.array([[values[0]], [values[0] + step]])
            for arr in (values, arr_2d):
                expected_arr = expected_real(arr, cfg['n_frac'], mode)
                x = Fxp(arr, rounding=mode, **cfg)
                assert np.allclose(x(), expected_arr)
                for alias in aliases:
                    xa = Fxp(arr, rounding=alias, **cfg)
                    assert np.allclose(xa(), expected_arr)
                    assert xa.config.rounding == mode

            if cfg['signed']:
                cvals_1d = np.array([
                    (0.5 * step) + 1j * (-0.5 * step),
                    (3.5 * step) + 1j * (0.1),
                    (-3.5 * step) + 1j * (-0.1),
                    0.1 + 1j * (-0.1),
                ])
                cvals_2d = np.vstack([cvals_1d, cvals_1d + (step + 1j * step)])
                for cvals in (cvals_1d, cvals_2d):
                    expected_c = expected_complex(cvals, cfg['n_frac'], mode)
                    xc = Fxp(cvals, rounding=mode, **cfg)
                    xv = xc()
                    xr = np.vectorize(lambda z: z.real)(xv)
                    xi = np.vectorize(lambda z: z.imag)(xv)
                    assert np.allclose(xr, expected_c.real)
                    assert np.allclose(xi, expected_c.imag)
                    for alias in aliases:
                        xca = Fxp(cvals, rounding=alias, **cfg)
                        xav = xca()
                        xar = np.vectorize(lambda z: z.real)(xav)
                        xai = np.vectorize(lambda z: z.imag)(xav)
                        assert np.allclose(xar, expected_c.real)
                        assert np.allclose(xai, expected_c.imag)


def test_rounding_nearest_posinf_comprehensive():
    """Validates nearest_posinf for ties/non-ties across scalar, array, complex, aliases, and >64-bit sizes."""

    def nearest_posinf_expected_real(values, n_frac):
        scale = 2 ** n_frac
        vals = np.asarray(values)
        return np.floor(vals * scale + 0.5) / scale

    def nearest_posinf_expected_complex(values, n_frac):
        vals = np.asarray(values)
        real_q = nearest_posinf_expected_real(vals.real, n_frac)
        imag_q = nearest_posinf_expected_real(vals.imag, n_frac)
        return real_q + 1j * imag_q

    # Issue #86 explicit reproducible case.
    x_issue = Fxp(0.1689453125, n_word=12, n_int=2, rounding='nearest_posinf')
    assert x_issue() == 0.169921875

    alias_modes = ['SC_RND', 'nearest_ties_to_posinf', 'roundTiesToPositive']
    for mode in alias_modes:
        x_alias = Fxp(0.1689453125, n_word=12, n_int=2, rounding=mode)
        assert x_alias.config.rounding == 'nearest_posinf'
        assert x_alias() == x_issue()

    configs = [
        dict(signed=True, n_word=8, n_frac=2),
        dict(signed=True, n_word=16, n_frac=9),
        dict(signed=True, n_word=80, n_frac=70),   # >64-bit path
        dict(signed=False, n_word=72, n_frac=64),  # >64-bit unsigned path
    ]

    for cfg in configs:
        step = 2.0 ** (-cfg['n_frac'])
        if cfg['signed']:
            values = np.array([
                0.0,
                0.5 * step,
                -0.5 * step,
                3.5 * step,
                -3.5 * step,
                10.2 * step,
                -10.2 * step,
                0.1,
                -0.1,
            ])
        else:
            values = np.array([
                0.0,
                0.5 * step,
                3.5 * step,
                10.2 * step,
                0.1,
                0.2,
            ])

        # Scalars
        for v in values:
            x = Fxp(float(v), rounding='nearest_posinf', **cfg)
            expected = nearest_posinf_expected_real(float(v), cfg['n_frac'])
            assert np.isclose(x(), expected)

        # Arrays: test 1D and 2D with multiple elements.
        arr_1d = values
        arr_2d = np.vstack([values, values + step]) if values.size > 1 else np.array([[values[0]], [values[0] + step]])

        for arr in (arr_1d, arr_2d):
            assert np.asarray(arr).size > 1
            x_arr = Fxp(arr, rounding='nearest_posinf', **cfg)
            expected_arr = nearest_posinf_expected_real(arr, cfg['n_frac'])
            assert np.allclose(x_arr(), expected_arr)

            # Aliases should match canonical mode output on arrays.
            for mode in alias_modes:
                x_arr_alias = Fxp(arr, rounding=mode, **cfg)
                assert np.allclose(x_arr_alias(), expected_arr)

        # Complex arrays (signed configs only): test 1D and 2D with multiple elements.
        if cfg['signed']:
            cvals_1d = np.array([
                (0.5 * step) + 1j * (-0.5 * step),
                (3.5 * step) + 1j * (10.2 * step),
                (-3.5 * step) + 1j * (-10.2 * step),
                0.1 + 1j * (-0.1),
            ])
            cvals_2d = np.vstack([cvals_1d, cvals_1d + (step + 1j * step)])

            for cvals in (cvals_1d, cvals_2d):
                assert np.asarray(cvals).size > 1
                x_c = Fxp(cvals, rounding='nearest_posinf', **cfg)
                expected_c = nearest_posinf_expected_complex(cvals, cfg['n_frac'])
                x_c_vals = x_c()
                x_c_real = np.vectorize(lambda z: z.real)(x_c_vals)
                x_c_imag = np.vectorize(lambda z: z.imag)(x_c_vals)
                assert np.allclose(x_c_real, expected_c.real)
                assert np.allclose(x_c_imag, expected_c.imag)

                for mode in alias_modes:
                    x_c_alias = Fxp(cvals, rounding=mode, **cfg)
                    x_c_alias_vals = x_c_alias()
                    x_c_alias_real = np.vectorize(lambda z: z.real)(x_c_alias_vals)
                    x_c_alias_imag = np.vectorize(lambda z: z.imag)(x_c_alias_vals)
                    assert np.allclose(x_c_alias_real, expected_c.real)
                    assert np.allclose(x_c_alias_imag, expected_c.imag)

def test_rounding_object_path_large_integer_passthrough():
    """Validates object-path passthrough for large integer values (>64-bit path)."""
    raw_vals = np.array([2**110 + 123, -(2**109) + 7, 2**108 + 1], dtype=object)
    modes = ['nearest_posinf', 'nearest_neginf', 'nearest_zero', 'nearest_away', 'bit_trunc', 'around', 'floor', 'ceil', 'fix', 'trunc']

    # raw=True always provides already-quantized integer values; rounding must be passthrough.
    for mode in modes:
        x = Fxp(None, signed=True, n_word=160, n_frac=0, rounding=mode)
        x.set_val(raw_vals, raw=True)
        out = x.raw()
        assert np.array_equal(np.array(out, dtype=object), raw_vals)

    # represented integer object arrays on >64-bit path should also remain unchanged for n_frac=0.
    repr_vals = np.array([2**90 + 5, -(2**89) + 11, 0, 1234567890123456789], dtype=object)
    for mode in modes:
        x = Fxp(repr_vals, signed=True, n_word=160, n_frac=0, rounding=mode)
        out = np.array(x.raw(), dtype=object)
        assert np.array_equal(out, repr_vals)


def test_rounding_object_array_multi_element_passthrough():
    """Validates multi-element ndarray(object) passthrough behavior for large integer values."""
    vals_1d = np.array([2**100 + 3, -(2**99) + 9, 2**98 + 1, -77], dtype=object)
    vals_2d = np.vstack([
        vals_1d,
        np.array([v + 2 for v in vals_1d], dtype=object),
    ])

    new_modes = ['nearest_posinf', 'nearest_neginf', 'nearest_zero', 'nearest_away', 'bit_trunc']

    for arr in (vals_1d, vals_2d):
        assert np.asarray(arr).size > 1

        # n_frac=0: passthrough should preserve represented integers.
        for mode in new_modes:
            x = Fxp(arr, signed=True, n_word=192, n_frac=0, rounding=mode)
            out = np.array(x.raw(), dtype=object)
            assert np.array_equal(out, arr)

        # aliases should follow the same passthrough behavior.
        x_sc_rnd = Fxp(arr, signed=True, n_word=192, n_frac=0, rounding='SC_RND')
        assert np.array_equal(np.array(x_sc_rnd.raw(), dtype=object), arr)

        x_sc_trn = Fxp(arr, signed=True, n_word=192, n_frac=0, rounding='SC_TRN')
        assert np.array_equal(np.array(x_sc_trn.raw(), dtype=object), arr)

        # n_frac>0: represented integers map to exact scaled raw integers, no rounding needed.
        n_frac = 7
        scale = 1 << n_frac
        expected_scaled = np.array(arr, dtype=object) * scale
        for mode in new_modes:
            x_scaled = Fxp(arr, signed=True, n_word=192, n_frac=n_frac, rounding=mode)
            out_scaled = np.array(x_scaled.raw(), dtype=object)
            assert np.array_equal(out_scaled, expected_scaled)

def test_rounding_warns_object_dtype_when_n_word_below_max():
    """Warns when object dtype reaches rounding despite n_word being below n_word_max."""
    x = Fxp(None, True, 8, 2, rounding='nearest_posinf')
    obj_vals = np.array([1.5, -1.5], dtype=object)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        _ = x._round(obj_vals, method='nearest_posinf')

    assert any(isinstance(w.message, RuntimeWarning) for w in caught)
    assert any('Object dtype reached rounding while n_word' in str(w.message) for w in caught)
