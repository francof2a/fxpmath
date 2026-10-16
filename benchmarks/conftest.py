"""Shared fixtures and metadata for fxpmath benchmarks (pytest-benchmark)."""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest

import fxpmath
from fxpmath import Fxp

SEED = 20261016
ARRAY_SIZE = 1000


def pytest_benchmark_update_machine_info(config, machine_info):
    """Add library versions to benchmark metadata so results are comparable."""
    # results are versioned in the repo: do not store the host name
    machine_info.pop('node', None)
    machine_info['numpy_version'] = np.__version__
    machine_info['fxpmath_version'] = fxpmath.__version__
    machine_info['fxpmath_n_word_max'] = fxpmath._n_word_max


@pytest.fixture
def rng():
    """Deterministic random generator for reproducible inputs."""
    return np.random.default_rng(SEED)


@pytest.fixture
def scalars():
    """Scalar Fxp operands in a regular (non extended) format."""
    return Fxp(1.25, True, 16, 8), Fxp(-0.75, True, 16, 8)


@pytest.fixture
def arrays(rng):
    """Array Fxp operands in a regular format; divisor values are kept away from zero."""
    x = Fxp(rng.uniform(-1, 1, ARRAY_SIZE), True, 16, 8)
    y = Fxp(rng.choice([-1, 1], ARRAY_SIZE) * rng.uniform(0.5, 1, ARRAY_SIZE), True, 16, 8)
    return x, y


@pytest.fixture
def arrays_extended(rng):
    """Array Fxp operands in extended precision (object raw storage)."""
    x = Fxp(rng.uniform(-1, 1, ARRAY_SIZE), True, 128, 96)
    y = Fxp(rng.choice([-1, 1], ARRAY_SIZE) * rng.uniform(0.5, 1, ARRAY_SIZE), True, 128, 96)
    return x, y


@pytest.fixture
def float_values(rng):
    """Float input values exceeding the s16/8 range, to exercise overflow handling."""
    return rng.uniform(-200, 200, ARRAY_SIZE)
