# generalized sizing

This chapter documents the generalized fixed-point sizing behavior supported by `Fxp`.

The core interpretation is:

`value = integer * 2**(-n_frac)`

Where:

* `integer` is the stored signed/unsigned integer represented with `n_word` bits.
* `n_frac` is scaling metadata (binary-point placement).

This means storage width (`n_word`) and scaling (`n_frac`) are intentionally decoupled.

## rationale

In many practical DSP and hardware workflows, fixed-point values are treated as:

* an integer container, plus
* a scale factor.

This allows format changes and reinterpretation without moving data, and supports intermediate representations used by quantized pipelines.

## sizing relations

`Fxp` derives integer bits (`n_int`) as:

* signed: `n_int = n_word - n_frac - 1`
* unsigned: `n_int = n_word - n_frac`

So unusual combinations can be valid, including negative `n_int`.

## supported edge cases

### 1) negative fractional bits (`n_frac < 0`)

Binary point is to the right of the LSB.

* effect: coarser resolution, larger step size
* common use: scaled accumulators and intermediate pipeline nodes

```python
from fxpmath import Fxp

x = Fxp(16, signed=True, n_word=8, n_frac=-2)
print(x.dtype, x.n_int, x())
# fxp-s8/-2 9 16.0
```

### 2) fractional bits greater than word bits (`n_frac > n_word`)

Binary point is to the left of the MSB.

* effect: very small representable magnitudes
* consequence: `n_int` becomes negative

```python
from fxpmath import Fxp

y = Fxp(-3, signed=True, n_word=1, n_frac=10)
print(y.dtype, y.n_int, y.raw(), y())
# fxp-s1/10 -10 -1 -0.0009765625
```

### 3) negative integer bits (`n_int < 0`)

This is the same geometric interpretation as case (2): binary point outside the stored word.

```python
from fxpmath import Fxp

z = Fxp(1, signed=False, n_word=4, n_frac=5)
print(z.dtype, z.n_int, z.raw(), z())
# fxp-u4/5 -1 15 0.46875
```

### 4) zero fractional bits (`n_frac = 0`)

Pure integer scaling in fixed-point form.

```python
from fxpmath import Fxp

w = Fxp(7, signed=True, n_word=8, n_frac=0)
print(w.dtype, w.n_int, w())
# fxp-s8/0 7 7.0
```

### 5) zero integer bits (`n_int = 0`)

Pure fractional-style range, commonly used for normalized signals and coefficients.

```python
from fxpmath import Fxp

u = Fxp(0.75, signed=True, n_word=8, n_frac=7)
print(u.dtype, u.n_int, u())
# fxp-s8/7 0 0.75
```

## signedness and overflow are orthogonal

Sizing only defines representable range and precision. Runtime behavior at limits depends on configuration:

* signed vs unsigned
* `overflow='wrap'` vs `overflow='saturate'`
* rounding mode

Example:

```python
from fxpmath import Fxp

a = Fxp(20.0, signed=False, n_word=4, n_frac=0, overflow='wrap')
b = Fxp(20.0, signed=False, n_word=4, n_frac=0, overflow='saturate')

print(a())  # wrapped modulo range
print(b())  # clipped to max
```

## practical guidance

* Use these generalized formats intentionally, especially when `n_int < 0` or `n_frac < 0`.
* Validate assumptions with `x.info(verbose=3)` to inspect limits and precision.
* Prefer explicit `dtype` strings in interfaces when sharing formats across modules.
