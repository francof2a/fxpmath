"""Benchmarks for Fxp creation and value assignment (set_val, rounding, overflow)."""

import pytest

from fxpmath import Fxp


@pytest.mark.benchmark(group='creation')
def test_create_scalar(benchmark):
    benchmark(Fxp, 1.25, True, 16, 8)


@pytest.mark.benchmark(group='creation')
def test_create_array(benchmark, float_values):
    benchmark(Fxp, float_values, True, 16, 8)


@pytest.mark.benchmark(group='creation')
def test_create_array_auto_sizing(benchmark, float_values):
    benchmark(Fxp, float_values)


@pytest.mark.benchmark(group='creation')
def test_create_like(benchmark, scalars):
    x, _ = scalars
    benchmark(Fxp, 0.5, like=x)


@pytest.mark.benchmark(group='set_val')
def test_set_val_array_saturate(benchmark, float_values):
    x = Fxp(None, True, 16, 8, overflow='saturate')
    benchmark(x.set_val, float_values)


@pytest.mark.benchmark(group='set_val')
def test_set_val_array_wrap(benchmark, float_values):
    x = Fxp(None, True, 16, 8, overflow='wrap')
    benchmark(x.set_val, float_values)


@pytest.mark.benchmark(group='set_val')
def test_set_val_array_around(benchmark, float_values):
    x = Fxp(None, True, 16, 8, rounding='around')
    benchmark(x.set_val, float_values)


@pytest.mark.benchmark(group='set_val')
def test_set_val_array_extended(benchmark, float_values):
    x = Fxp(None, True, 128, 96)
    benchmark(x.set_val, float_values)


@pytest.mark.benchmark(group='set_val')
def test_set_val_array_raw(benchmark, arrays):
    x, _ = arrays
    raw = x.val.copy()
    y = Fxp(None, True, 16, 8)
    benchmark(y.set_val, raw, raw=True)


@pytest.mark.benchmark(group='set_val')
def test_get_val_array(benchmark, arrays):
    x, _ = arrays
    benchmark(x.get_val)
