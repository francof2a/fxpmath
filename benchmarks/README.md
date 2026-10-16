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

| Benchmark | NumPy 2.5.4 | NumPy 1.26.4 |
|---|---|---|
| `create_scalar` | 0.20 ms | 0.25 ms |
| `create_array` (1k) | 1.40 ms | 1.39 ms |
| `create_array_auto_sizing` (1k) | 33.2 ms | 71.7 ms |
| `set_val_array_saturate` (1k) | 1.27 ms | 1.23 ms |
| `set_val_array_wrap` (1k) | 0.09 ms | 0.14 ms |
| `scalar[add]` / `scalar[mul]` | 0.30 / 0.32 ms | 0.36 / 0.37 ms |
| `scalar_add_constant` | 0.62 ms | 0.73 ms |
| `array[add]` / `array[mul]` (1k) | 1.44 / 1.44 ms | 1.43 / 1.46 ms |
| `extended[add]` / `extended[mul]` (1k) | 1.89 / 1.93 ms | 2.00 / 2.04 ms |
| `dispatch_clip` (1k) | 3.90 ms | 3.76 ms |
| `utils_clip_float` (1k) | 1.07 ms | 0.93 ms |
| `setitem_loop_2d` (384 updates) | 402 ms | 482 ms |

Main findings: saturation (`utils.clip` through `np.vectorize`) is about 93% of `set_val` cost on 1k arrays and dominates array operators; scalar operators are dominated by `Fxp` construction overhead; auto sizing creation is about 25x slower than explicit sizing.
