"""Benchmarks for arithmetic hot paths (add, sub, mul, truediv, floordiv, pow) by operand kind."""

import operator

import pytest

from fxpmath import Fxp

OPS = {
    'add': operator.add,
    'sub': operator.sub,
    'mul': operator.mul,
    'truediv': operator.truediv,
    'floordiv': operator.floordiv,
}


@pytest.mark.benchmark(group='ops-scalar')
@pytest.mark.parametrize('op', OPS)
def test_scalar(benchmark, scalars, op):
    x, y = scalars
    benchmark(OPS[op], x, y)


@pytest.mark.benchmark(group='ops-scalar')
def test_scalar_pow(benchmark, scalars):
    x, _ = scalars
    e = Fxp(2, True, 8, 0)
    benchmark(operator.pow, x, e)


@pytest.mark.benchmark(group='ops-scalar')
def test_scalar_add_constant(benchmark, scalars):
    x, _ = scalars
    benchmark(operator.add, x, 0.5)


@pytest.mark.benchmark(group='ops-array')
@pytest.mark.parametrize('op', OPS)
def test_array(benchmark, arrays, op):
    x, y = arrays
    benchmark(OPS[op], x, y)


@pytest.mark.benchmark(group='ops-array')
def test_array_pow(benchmark, arrays):
    x, _ = arrays
    e = Fxp(2, True, 8, 0)
    benchmark(operator.pow, x, e)


@pytest.mark.benchmark(group='ops-array')
def test_array_add_constant(benchmark, arrays):
    x, _ = arrays
    benchmark(operator.add, x, 0.5)


@pytest.mark.benchmark(group='ops-extended')
@pytest.mark.parametrize('op', ['add', 'mul', 'truediv'])
def test_extended(benchmark, arrays_extended, op):
    x, y = arrays_extended
    benchmark(OPS[op], x, y)
