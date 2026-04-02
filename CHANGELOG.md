# Changelog

All notable changes to this project will be documented in this file.

The format is inspired by [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), adapted to fit `fxpmath` release history and compatibility tracking.

## [0.5.0]

### Compatibility
- Removed default import-time mutation of `NUMPY_EXPERIMENTAL_ARRAY_FUNCTION` and fixed its environment parsing bug.
- Defined support policy for Python `3.9` to `3.14` and NumPy `1.26.4` to `<3` in packaging metadata and dependency files.
- Expanded CI and release-gate workflows with explicit NumPy `1.26.4` and latest `2.x` matrix lanes across valid Python versions.
- Normalized `out=...` handling across operator helpers so NumPy ellipsis is treated as an omitted output argument.
- Kept strict `Fxp`-only `out` contract for fxpmath-managed dispatch and explicit errors for invalid `out` and `out_like` types.
- Updated `_wrapped_numpy_func` output routing to make `__array_ufunc__` and `__array_function__` paths consistent for `out` and `out_like` handling.
- Modernized NumPy protocol hooks: `Fxp.__array__` now supports NumPy `2.x` `dtype` and `copy` behavior while preserving legacy positional compatibility.
- Removed legacy `__array_prepare__` hook and extended `Fxp.__array_wrap__` with `return_scalar` support for NumPy `2.x` compatibility.

### Added
- Added isolated import regression coverage to verify `import fxpmath` preserves absent, `0`, and `1` states for `NUMPY_EXPERIMENTAL_ARRAY_FUNCTION`.
- Added mixed-input NumPy divide regression coverage.
- Added focused feature tests for `out=...`, scalar and 0-D output paths, strict output-type validation, and array protocol compatibility in NumPy-facing and function-level test files.
- Added user-facing compatibility notes and NumPy migration guidance in `docs/compatibility_notes.md`.
- Started internal module reorganization for `0.5.0` with `fxpmath/operators/` split into `core`, `arithmetic`, `reduction`, and `array`.
- Extracted `Config` from `fxpmath/objects.py` into `fxpmath/config.py`.

### Changed
- Narrowed fallback conversion retries in `_wrapped_numpy_func` so unsupported kwargs raise cleanly instead of being retried through broad implicit coercion.
- Converted `fxpmath/functions.py` into a backward-compatible public facade re-exporting operator functions from `fxpmath.operators`.
- Converted monolithic `fxpmath/utils.py` into a package-based layout under `fxpmath/utils/` with `common`, `parse`, `repr`, `bitwise`, and `numeric` modules.
- Preserved compatibility for `from fxpmath import utils`, `from fxpmath import Config`, and `from fxpmath.objects import Config` during the reorganization.
- Kept NumPy dispatch registrations and public operation signatures stable during the refactor.

### Documentation
- Documented supported Python and NumPy ranges plus finalized NumPy dispatch output behavior.
- Moved compatibility guidance out of the README into dedicated documentation.

### Validation
- Validated targeted compatibility test groups on NumPy `1.26.4` and latest `2.x`.
- Added stable CI gate jobs for required NumPy compatibility and installation smoke checks, plus artifact uploads for lint/test traceability.
- Added a scheduled nightly workflow against NumPy pre-release wheels.
- Ran the full suite successfully on the primary environment (`141 passed, 1 warning`).

## [0.4.10]

### Added
- Added installation smoke tests in `tests/test_installation.py` to create an isolated virtual environment, install from source, and verify import path and version.
- Added regression test coverage for complex `uraw` conversion parity (issue `#102`).
- Added regression coverage for repeated `int()` casting across Linux sizing paths (issue `#98`).
- Added `Fxp.reshape_inplace(shape, order='C')` as the explicit in-place reshape API.

### Changed
- Modernized packaging to PEP `517` and PEP `621` using `pyproject.toml` as canonical metadata source.
- Added dynamic versioning from `fxpmath.__version__` and setuptools package discovery via `find`.
- Cleaned up development-plan tracking by removing a duplicated issue `#86` status entry.
- Deprecated `Fxp.shape = ...` assignment with `DeprecationWarning`; prefer `reshape()` or `reshape_inplace()`.

### Fixed
- Fixed complex `uraw` conversion by applying unsigned two's-complement independently to real and imaginary components (issue `#102`).
- Fixed scalar integer casting after repeated in-place increments by using scalar-safe integer conversion in `astype` and object-cast `set_val` paths (issue `#98`).
- Defined complex bitwise semantics for issue `#55`: mixed complex and non-complex operands now warn once and apply the real operand to both components, while complex-complex applies operations part-wise.
- Refactored bitwise internals to support array-vs-array operands with NumPy broadcasting semantics for `&`, `|`, `^`, and `~`.
- Aligned `Fxp.reshape` with NumPy by returning a reshaped copy instead of mutating the source object.
- Improved reshape compatibility by using positional ndarray reshape arguments across NumPy versions.

### Tests
- Kept issue `#55` regression minimal in `tests/test_issues.py` and moved broader behavior coverage to `tests/test_complex_numbers.py`.
- Expanded bitwise operator coverage in `tests/test_operators.py` for multidimensional arrays, broadcast paths, scalar masks, and `>64`-bit cases.

## [0.4.9]

### Added
- Added `from_bin` method and function for creating or setting `Fxp` values from binary strings (issue `#49`).
- Added support for complex binary strings as input format.
- Added prefix selection for binary and hexadecimal representations.

### Changed
- Forced `config.op_input_size = 'best'` when power uses a constant non-`Fxp` operand and added a warning (issue `#89`).

### Fixed
- Fixed wrap behavior beyond `n_word_max` (issue `#41`).
- Propagated the inaccuracy flag to new or resulting `Fxp` objects when any input is inaccurate (issue `#48`).
- Fixed `cumsum` for sizes larger than `32` bits on Windows and `64` bits on Linux (issue `#76`).
- Fixed `numpy.reshape` handling so the default sizing remains `same` instead of `optimal` (issue `#77`).
- Fixed negative number parsing in dtype strings (issue `#80`).

## [0.4.8]

### Fixed
- Fixed value dtype handling for Windows and `uint` values treated as `32` bits.
- Updated `__getitem__` so it returns an `Fxp` with raw value as a view, solving issue `#62`.

### Tests
- Added tests for issues `#60` and `#62`.

## [0.4.7]

### Added
- Added `abs` method.

### Changed
- Kept `val` as the original `vdtype` instead of forcing `object` when it is safe to do so.
- Converted `vdtype` to `float` when scaling transformation is applied.

## [0.4.6]

### Fixed
- Fixed complex `truediv` and `floordiv` methods (issue `#53`).
- Fixed complex binary, hexadecimal, and base representations for arrays (issue `#56`).
- Fixed dtype-based initialization when the dtype is complex (issue `#58`).

### Tests
- Updated tests for complex representations.

## [0.4.5]

### Added
- Added `dtype` argument to `resize`.
- Added `get_dtype` method with `notation` argument.
- Added docstring updates for `get_dtype`, `resize`, `reshape`, `flatten`, `set_val`, `astype`, `get_val`, `raw`, `uraw`, and `equal`.
- Added `order` parameter support for `reshape` and `flatten`.

### Fixed
- Fixed `FutureWarning` in subdtype `'str'` comparison (issue `#45`).

## [0.4.4]

### Fixed
- Fixed wrapping and scaling behavior (issue `#44`).

## [0.4.3]

### Added
- Added optional `config` init parameter to `Fxp`.
- Added `copy` and `deepcopy` methods to `Config`.
- Linked `functions.truediv` with `numpy.divide` dispatch.
- Added `axes` parameter to `transpose`.

### Changed
- Functions operating on one or two values now copy the first operand configuration by default.

### Fixed
- Fixed wrap overflow after operations (issue `#42`).
- Fixed `utils.wrap` when data exceeds `n_word_max` (issue `#41`).
- Fixed warning about internal complex checks (issue `#39`).

## [0.4.2]

### Added
- Added template support to `Config`.

### Fixed
- Fixed config update behavior from an `Fxp` template.

## [0.4.1]

### Added
- Added NumPy precision testing at init.
- Defined `n_word_max` and config error handling independently.
- Added support for NumPy arrays of strings.

### Fixed
- Fixed `astype` casting when `dtype` and `vdtype` are `None`.
- Fixed casting to `object` in extended `cumprod`, extended precision multiply, and constructor paths.
- Fixed integer representation behavior.
- Fixed issues `#26` and `#31`.

## [0.4.0]

### Added
- Added extended precision support beyond `64` bits for `add`, `sub`, `mult`, `truediv`, `floordiv`, `mod`, and `pow`.
- Added support for `tuple` and `Decimal` input formats.
- Added dtype-based initialization for `fxp` and `Q` formats.
- Added template functionality for defining default formats for new `Fxp` objects.
- Added `Config` class to manage low-level object behavior.
- Added extended precision status flag (`extended_prec`).
- Added new `Fxp` properties: `size`, `shape`, `ndim`, and `T`.
- Added new `Fxp` methods: `reshape`, `flatten`, `raw`, and `uraw`.
- Added NumPy-array-style methods on `Fxp`: `all`, `any`, `argmax`, `argmin`, `argpartition`, `argsort`, `max`, `min`, `mean`, `std`, `var`, `sum`, `cumsum`, `cumprod`, `ravel`, `tolist`, `sort`, `conjugate`, `conj`, `transpose`, `item`, `clip`, `diagonal`, `trace`, `prod`, `dot`, and `nonzero`.
- Added support for `bool`, `int`, `float`, and `complex` casting.
- Added dtype information to `Fxp.__repr__`.
- Added basic callbacks: `on_value_change`, `on_status_overflow`, `on_status_underflow`, and `on_status_inaccuracy`.
- Added NumPy function dispatch so `Fxp` can be used as both NumPy inputs and outputs.

### Changed
- Migrated several core calculations and methods from `float` to `int` to support extended precision.
- Results of ALU operations with constants now keep the same size as the `Fxp` object by default.
- `like` initialization now supports overwriting size parameters.
- Improved performance in several core paths.

### Notes
- NumPy `>= 1.17.x` was recommended to enable function dispatch by default.
- With older NumPy versions, users were advised to import `fxpmath` before NumPy or set `NUMPY_EXPERIMENTAL_ARRAY_FUNCTION=1` before importing NumPy.

## [0.3.9]

### Fixed
- Fixed wrapping behavior (issue `#20`).

## [0.3.8]

### Added
- Added padding support for hexadecimal representations.

### Fixed
- Fixed bug with `'b'` in hexadecimal strings (issue `#15`).
- Fixed spelling/wording of `inaccuracy` in status (issue `#16`).
- Fixed `__pow__` behavior (issue `#17`).
- Fixed bug where indexed complex assignment stored only the imaginary part (issue `#19`).

## [0.3.7]

### Added
- Added right-shift support for `shifting='expand'` mode.
- Added `sum` function in `functions`.
- Added support for `Fxp` words and values larger than `64` bits.

### Changed
- Improved performance in indexed assignment.
- `Fxp` slicing now returns an `Fxp` object when values are arrays.
- `Fxp` now accepts another `Fxp` as input value.
- Updated docstrings.

## [0.3.6]

### Fixed
- Fixed `setup.py` dependency handling so package dependencies install correctly.

## [0.3.5]

### Fixed
- Fixed wrap error for ndarrays.

## [0.3.4]

### Fixed
- Fixed wrap error for unsigned `Fxp`.

## [0.3.3]

### Fixed
- Fixed bitwise shifting of unsigned values.
- Fixed wrap error when the input is a float.

## [0.3.2]

### Fixed
- Fixed support for N-dimensional array inputs (`N > 1`).

## [0.3.1]

### Fixed
- Fixed initialization and `best_size_calc` behavior for Win32 platforms.

## [0.3.0]

### Added
- Added logical (bitwise) operators.
- Added binary shifting.
- Added value scaling (scale and offset).
- Added comparison support.
- Added inaccuracy status flag to indicate stored value differs from input.
- Added verbosity support to `info()`.
- Added `reset()` to clear status flags.
- Added raw-value support to `set_val()`.
- Added power operator support.
- Added inplace operator support.

### Fixed
- Fixed several bugs.

## [0.2.0]

### Added
- Added binary, hexadecimal, and base representations.
- Added upper, lower, and precision properties.
- Added arithmetic operator support.
- Added support for string inputs including binary, hexadecimal, and fractional formats.
- Added `Fxp` copying.
- Added initialization from another `Fxp`.
- Added indexing support.

## [0.1.0]

### Added
- First public release.
