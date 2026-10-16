---
title: 'fxpmath: fixed-point arithmetic emulation in Python'
tags:
  - Python
  - fixed-point arithmetic
  - NumPy
  - FPGA
  - digital signal processing
  - quantization
  - hardware modeling
authors:
  - name: Franco Alberto Alcaraz
    orcid: 0009-0003-8330-6223
    affiliation: 1
affiliations:
  - name: TODO (affiliation)
    index: 1
date: TODO
bibliography: paper.bib
---

<!--
DRAFT for JOSS submission, planned after the v0.5.0 release (ROADMAP.md, M6).
Target length: 750-1750 words (body). Items marked TODO must be completed before submitting.
-->

# Summary

Digital hardware such as FPGAs, ASICs and many microcontrollers often represents real numbers in fixed point: an integer word of fixed width whose bits are interpreted with an implicit binary point. Word width, signedness, rounding and overflow handling determine the numerical behavior of the final design. Designers therefore need a model of the algorithm that reproduces the hardware arithmetic bit by bit before writing HDL or high-level synthesis code.

`fxpmath` is a Python library for fractional fixed-point (base 2) arithmetic and binary manipulation. Its `Fxp` object holds scalars or N-dimensional arrays with an arbitrary word size, fractional size and signedness. It supports arithmetic, bitwise and comparison operators, configurable rounding and overflow behavior, status flags and binary and hexadecimal representations. `Fxp` objects interoperate with NumPy [@harris2020array], so existing array code can be evaluated in fixed point with few changes.

# Statement of need

Fixed-point design usually starts from a floating-point reference written in Python or MATLAB. The commercial reference tool for the conversion step is the MATLAB Fixed-Point Designer [@matlab_fixedpoint], which is proprietary. Python is now the common language for algorithm prototyping, machine learning and hardware verification (for example with cocotb), so researchers and engineers need an open tool that answers questions like these:

- How many integer and fractional bits does a signal need?
- How much error does a given rounding mode introduce?
- Does an intermediate result overflow, and with what effect?
- What exact bit pattern should the hardware produce for a given input?

`fxpmath` targets researchers and engineers in digital signal processing, FPGA and ASIC design, quantized neural network inference and embedded systems. It lets them explore word sizes, check a quantized model against a floating-point reference and generate bit-exact test vectors for hardware simulations, without leaving the Python and NumPy ecosystem.

# State of the field

Several Python packages emulate fixed-point arithmetic [@apytypes_comparison]:

- `spfpm` [@spfpm] provides scalar fixed-point numbers with strict overflow checking and elementary functions, but no array support.
- `fpbinary` [@fpbinary] is a C extension that follows the semantics of the VHDL fixed-point package, also scalar-only.
- `fixedpoint` [@fixedpoint_sel] offers well-documented scalar fixed-point objects.
- `numfi` [@numfi] subclasses `numpy.ndarray` and is limited to 64-bit words, including intermediate results.
- Quantization packages such as `pyfixp` and `pychop` quantize arrays of floating-point values but compute at the precision of the underlying array, so results are not bit-exact for long words.
- APyTypes [@henriksson2024apytypes] provides fixed- and floating-point scalar and array types with a C++ core, focused on simulation speed.

`fxpmath` was first released in 2020 and was among the first packages to combine N-dimensional NumPy arrays, arbitrary word lengths (including words longer than 64 bits) and configurable rounding, overflow and result sizing in a single object. Its design favors exact, bit-true results and NumPy-style ergonomics over raw speed. APyTypes is faster for large simulations; `fxpmath` remains pure Python on top of NumPy, which keeps installation trivial on any platform and lets users mix `Fxp` objects with NumPy functions directly.

TODO: add a short summary of the comparative benchmark against NumPy, APyTypes and numfi (ROADMAP M5), stating the measured trade-off.

# Software design

An `Fxp` object stores its value as raw integers together with its format (signedness, word size `n_word` and fractional size `n_frac`). The real value is `raw / 2**n_frac`. All rounding, overflow and bitwise operations act on the raw integers, so results are exact and reproducible.

- **Storage and extended precision.** Raw values live in NumPy `int64`/`uint64` arrays. When the word size reaches the platform integer width, `fxpmath` switches to object arrays of Python integers, which have unlimited precision. This path supports words of hundreds of bits at a higher cost per element. `decimal.Decimal` is supported as input and output to carry values beyond float precision.
- **Quantization behavior.** Ten rounding modes (for example `trunc`, `around`, `floor`, `ceil`, `fix` and several round-to-nearest variants) and two overflow modes (`saturate` and `wrap`) are available. Status flags record overflow, underflow and inaccuracy events.
- **Result sizing.** A per-object `Config` controls how the format of an operation result is chosen. The `op_sizing` option selects between growing the result to avoid information loss (`optimal`, the default), keeping the operand format (`same`), and other policies (`fit`, `largest`, `smallest`). The same mechanism controls operations with constants, the arithmetic method (on raw integers or on the represented values) and whether array results are returned as `Fxp` objects or NumPy arrays.
- **NumPy integration.** `Fxp` implements the NumPy dispatch protocols `__array_ufunc__` and `__array_function__`. A registry maps NumPy functions to fixed-point implementations, so calls such as `np.sum`, `np.cumsum`, `np.dot` or `np.clip` on `Fxp` inputs return `Fxp` results with the configured sizing. Unregistered functions fall back to the represented values.
- **Representation.** Values can be created from and converted to integers, floats, complex numbers, strings in binary, hexadecimal or decimal, and `Decimal`. Formats can be written as `fxp-s16/8` or in Q notation. A linear scale and bias can be attached for non-power-of-two ranges.

The following example quantizes an array to a signed 8-bit format with 5 fractional bits, inspects the bit patterns and detects an overflow:

```python
from fxpmath import Fxp

x = Fxp([0.75, -1.3, 2.1], signed=True, n_word=8, n_frac=5)
x.bin(frac_dot=True)   # ['000.11000', '110.10111', '010.00011']

y = x * x              # full-precision product: fxp-s16/10
z = Fxp(4.2, like=x)   # saturates to 3.96875
z.status['overflow']   # True
```

The main trade-off of this design is speed. Every operation creates a new object, validates its inputs and applies quantization in Python, which adds overhead compared to plain NumPy, especially for element-by-element loops. The v0.5.0 release added a benchmark suite and vectorized the overflow and saturation paths, which made array operators about 4.5 times faster than in earlier versions.

TODO: update this section after the v0.5.0 core refactor (ROADMAP M3) and matrix multiplication support (M4).

# Research impact statement

`fxpmath` is used in published research and in hardware design workflows:

- pyrtlnet, an end-to-end Python workflow for quantized neural network inference hardware, uses `fxpmath` in its NumPy reference implementation of integer-only inference, which matches the PyRTL hardware model exactly [@sherwood2026pyrtlnet].
- @lomasbarrie2022multiview used it to represent fixed-point values and estimate quantization error in an FPGA implementation of a multiview object recognition network.
- @noyadominguez2026numerical used it to emulate fixed-point arithmetic in a study of the numerical stability of mixed-precision mutual information feature selection.
- @paul2024attacking selected it as the fixed-point library for experiments on attacks against machine learning inference in secure multiparty computation.
- TODO: confirm how BRAIN-AD [@tenwar2026brainad] uses `fxpmath` before citing it here.

The package is downloaded from PyPI about 120,000 times per month (TODO: update at submission). It is recommended in the AMD/Xilinx PYNQ community forum as a Python counterpart of the HLS `ap_fixed` types and is listed among the fixed-point resources in the cocotb wiki. Its GitHub repository has more than 200 stars and has received issues and pull requests from many external users.

# AI usage disclosure

TODO: complete and review before submitting. Generative AI (Claude, Anthropic) was used during the v0.5.0 development cycle to assist with code changes, tests, benchmarks and documentation, and to prepare a first draft of this paper. All AI-assisted changes were reviewed by the author, merged through pull requests and validated with the test suite. The author reviewed and edited the paper text and checked every reference.

# Acknowledgements

TODO: acknowledge contributors and any financial support (or state that there was none).

# References
