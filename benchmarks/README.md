# fxpmath benchmarks

Repeatable performance benchmarks for fxpmath hot paths, based on [pytest-benchmark](https://pytest-benchmark.readthedocs.io/).
They are excluded from the default test run (`testpaths = ["tests"]` in `pyproject.toml`).

## Groups

| Group | What it measures |
|---|---|
| `creation` | `Fxp` construction: scalar, array, auto sizing, `like` |
| `set_val` | value assignment: rounding, saturation, wrap, extended precision, raw input, `get_val` |
| `ops-scalar` | scalar arithmetic (`add`, `sub`, `mul`, `truediv`, `floordiv`, `pow`, constant operand) |
| `ops-array` | same operators over 1000-element arrays |
| `ops-extended` | `add`, `mul`, `truediv` in extended precision (`s128/96`, object raw storage) |
| `numpy-dispatch` | registered NumPy functions (`sum`, `cumsum`, `max`, `sort`, `dot`, `clip`) |
| `helpers` | internal `utils.clip` / `utils.wrap` |
| `indexing` | `__getitem__` and element-wise `__setitem__` loop |

Inputs use a fixed random seed (`benchmarks/conftest.py`).

## Running

Install dev dependencies (`pip install -r requirements-dev.txt`), then:

```bash
# run and print results
pytest benchmarks

# quick check that all benchmarks run (no timing)
pytest benchmarks --benchmark-disable

# save a run, then compare a later run against it and fail on regressions
pytest benchmarks --benchmark-autosave
pytest benchmarks --benchmark-compare --benchmark-compare-fail=median:20%
```

Timings depend on the machine: compare runs from the same machine and environment only.
On the baseline platform, run-to-run noise was about 3% median and up to 9% max, so use a regression threshold of at least 20%.

## Baseline (v0.5.0 development)

### Platform

| Item | Value |
|---|---|
| OS | Ubuntu 22.04.5 LTS (Linux 6.8, x86_64) |
| CPU | Intel Core i7-8750H @ 2.20 GHz (6 cores, 12 threads) |
| Memory | 16 GB |
| Python | CPython 3.12.1 |
| NumPy | 2.5.4 and 1.26.4 |
| fxpmath | 0.5.0-dev2 (commit `b418e8e`) |

### Results (median time)

Baseline is the v0.5.0 development state before optimizations (commit `b418e8e`). Each optimization step adds a column.

**NumPy 2.5.4**

| Benchmark | Baseline | Saturation |
|---|---|---|
| `create_scalar` | 0.20 ms | 0.18 ms |
| `create_array` (1k) | 1.40 ms | 0.23 ms |
| `create_array_auto_sizing` (1k) | 33.2 ms | 30.6 ms |
| `set_val_array_saturate` (1k) | 1.27 ms | 0.10 ms |
| `set_val_array_raw` (1k) | 1.18 ms | 0.08 ms |
| `set_val_array_wrap` (1k) | 0.09 ms | 0.10 ms |
| `scalar[add]` / `scalar[mul]` | 0.30 / 0.32 ms | 0.27 / 0.29 ms |
| `scalar_add_constant` | 0.62 ms | 0.57 ms |
| `array[add]` / `array[mul]` (1k) | 1.44 / 1.44 ms | 0.31 / 0.32 ms |
| `array_pow` (1k) | 2.56 ms | 1.44 ms |
| `extended[add]` / `extended[mul]` (1k) | 1.89 / 1.93 ms | 1.87 / 1.91 ms |
| `dispatch_unary[cumsum]` (1k) | 1.39 ms | 0.26 ms |
| `dispatch_clip` (1k) | 3.90 ms | 2.67 ms |
| `setitem_loop_2d` (384 updates) | 402 ms | 379 ms |

**NumPy 1.26.4**

| Benchmark | Baseline | Saturation |
|---|---|---|
| `create_scalar` | 0.25 ms | 0.23 ms |
| `create_array` (1k) | 1.39 ms | 0.30 ms |
| `create_array_auto_sizing` (1k) | 71.7 ms | 65.1 ms |
| `set_val_array_saturate` (1k) | 1.23 ms | 0.15 ms |
| `set_val_array_raw` (1k) | 1.16 ms | 0.12 ms |
| `set_val_array_wrap` (1k) | 0.14 ms | 0.15 ms |
| `scalar[add]` / `scalar[mul]` | 0.36 / 0.37 ms | 0.34 / 0.35 ms |
| `scalar_add_constant` | 0.73 ms | 0.71 ms |
| `array[add]` / `array[mul]` (1k) | 1.43 / 1.46 ms | 0.39 / 0.41 ms |
| `array_pow` (1k) | 2.59 ms | 1.54 ms |
| `extended[add]` / `extended[mul]` (1k) | 2.00 / 2.04 ms | 1.93 / 1.96 ms |
| `dispatch_unary[cumsum]` (1k) | 1.40 ms | 0.34 ms |
| `dispatch_clip` (1k) | 3.76 ms | 2.76 ms |
| `setitem_loop_2d` (384 updates) | 482 ms | 462 ms |

### Optimization steps

1. **Saturation**: `np.vectorize`-based clip in the overflow path replaced by vectorized `utils.saturate`/`utils.overflow_masks` with exact integer bounds. `set_val` on 1k arrays about 12x faster and array operators about 4.5x faster. `set_val` with wrap is about 9 µs slower (exact overflow flags), within the regression threshold.

### Findings and next targets

- Scalar operators are dominated by `Fxp` construction overhead (copies, repeated `set_val`), not by arithmetic.
- After the saturation step, fxpmath is still about 9x slower than plain NumPy doing the same quantization or raw addition on 1M-element arrays: the gap comes from redundant passes over data in the core (pre-scans, casts, inaccuracy check, recomputing represented values), not from arithmetic.
- Extended precision (more than 64 bits) uses object arrays with Python ints, about 1.7 µs per element for an addition; this is the main candidate for a compiled engine.
- Auto sizing creation is about 100x slower than explicit sizing.
