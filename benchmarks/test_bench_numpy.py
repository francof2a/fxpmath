"""Benchmarks for NumPy dispatch, internal helpers (clip, wrap) and indexing."""

import numpy as np
import pytest

from fxpmath import Fxp, utils


@pytest.mark.benchmark(group='numpy-dispatch')
@pytest.mark.parametrize('func', [np.sum, np.cumsum, np.max, np.sort])
def test_dispatch_unary(benchmark, arrays, func):
    x, _ = arrays
    benchmark(func, x)


@pytest.mark.benchmark(group='numpy-dispatch')
def test_dispatch_dot(benchmark, arrays):
    x, y = arrays
    benchmark(np.dot, x, y)


@pytest.mark.benchmark(group='numpy-dispatch')
def test_dispatch_clip(benchmark, arrays):
    x, _ = arrays
    benchmark(np.clip, x, -0.5, 0.5)


@pytest.mark.benchmark(group='helpers')
def test_utils_clip_float(benchmark, float_values):
    benchmark(utils.clip, float_values, -128.0, 127.0)


@pytest.mark.benchmark(group='helpers')
def test_utils_clip_object(benchmark, float_values):
    vals = (float_values * 2**100).astype(object)
    benchmark(np.clip, vals, -2**106, 2**106 - 1)


@pytest.mark.benchmark(group='helpers')
def test_utils_wrap(benchmark, float_values):
    benchmark(utils.wrap, float_values, True, 8)


@pytest.mark.benchmark(group='indexing')
def test_getitem(benchmark, arrays):
    x, _ = arrays
    benchmark(x.__getitem__, 10)


@pytest.mark.benchmark(group='indexing')
def test_setitem_loop_2d(benchmark, rng):
    a = Fxp(rng.uniform(size=(24, 16)))
    step = 0.1

    def update():
        for i in range(24):
            for j in range(16):
                a[i, j] = a[i, j] + step

    benchmark.pedantic(update, rounds=5, iterations=1, warmup_rounds=1)
