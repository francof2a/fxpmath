
<img src="./docs/figs/fxpmath_logotipo.png" width="300">

A python library for fractional fixed-point (base 2) arithmetic and binary manipulation with Numpy compatibility.

Some key features:

* Fixed-point signed and unsigned numbers representation.
* Arbitrary word and fractional sizes. Auto sizing capability. Extended precision capability.
* Arithmetic and logical (bitwise) operations supported.
* Input values can be: int, float, complex, list, numpy arrays, strings (bin, hex, dec), Decimal type.
* Input rounding methods, overflow and underflow behaviors and flags.
* Binary, Hexadecimal, and other bases representations (like strings).
* Indexing supported.
* Linear scaling: scale and bias.
* Numpy backend.
* Suppport for Numpy functions. They can take and return Fxp objects.
* Internal behavior configurable: inputs/outputs formating, calculation methods.

visit [documentation](https://francof2a.github.io/fxpmath/) for more information.

See some examples in the [examples folder](https://github.com/francof2a/fxpmath/tree/master/examples).

---

![GitHub](https://img.shields.io/github/license/francof2a/fxpmath?style=for-the-badge)

![GitHub tag (latest SemVer)](https://img.shields.io/github/v/tag/francof2a/fxpmath?style=for-the-badge)

![PyPI](https://img.shields.io/pypi/v/fxpmath?style=for-the-badge)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/fxpmath?style=for-the-badge)

![Conda](https://img.shields.io/conda/v/francof2a/fxpmath?style=for-the-badge)
![Conda](https://img.shields.io/conda/pn/francof2a/fxpmath?style=for-the-badge)

---

## Table of content

* [install](#install)
* [citation](#citation)
* [quick start](#quick-start)
  * [creation](#creation)
  * [Representations](#representations)
  * [changing values](#changing-values)
  * [changing size](#changing-size)
  * [data types supported](#data-types-supported)
  * [indexing](#indexing)
* [arithmetic](#arithmetic)
* [logical (bitwise) operators](#logical--bitwise--operators)
* [Comparisons](#comparisons)
* [behaviors](#behaviors)
  * [overflow / underflow](#overflow---underflow)
  * [rounding](#rounding)
  * [inaccuracy](#inaccuracy)
* [Status flags](#status-flags)
* [copy](#copy)
* [Scaling](#scaling)

---

## install

To install from *pip* just do the next:

```bash
pip install fxpmath
```

To install with *conda* just do the next:

```bash
conda install -c francof2a fxpmath
```

Or you can clone the repository doing in your console:

```bash
git clone https://github.com/francof2a/fxpmath.git
```

and then go to the fxpmath folder and install it:

```bash
cd fxpmath
pip install .
```

## citation

```
Alcaraz, F., 2020. Fxpmath. Available at: https://github.com/francof2a/fxpmath.
```

**BibTex format**:

```
@online{alcaraz2020fxpmath,
  title={fxpmath},
  author={Alcaraz, Franco},
  year={2020},
  publisher={GitHub},
  url={https://github.com/francof2a/fxpmath},
}
```

## quick start

### creation

Let's jump into create our new **fractional fixed-point** variable:

```python
from fxpmath import Fxp

x = Fxp(-7.25)      # create fxp variable with value 7.25
x.info()
```

> dtype           =       fxp-s6/2  
> Value           =       -7.25

We have created a variable of 6 bits, where 1 bit has been reserved for sign, 2 bits for fractional part, and 3 remains for integer part. Here, bit sizes had been calculated to just satisfy the value you want to save.

But the most common way to create a **fxp** variable beside the its value is defining explicitly if it is signed, the number of bits for whole word and for the fractional part.

**Note**: `dtype` of Fxp object is a propietary *type* of each element stored in it. The format is:

**`fxp-<sign><n_word>/<n_frac>-{complex}`**

i.e.: `fxp-s16/15`, `fxp-u8/1`, `fxp-s32/24-complex`

```python
x = Fxp(-7.25, signed=True, n_word=16, n_frac=8)
```

or just

```python
x = Fxp(-7.25, True, 16, 8)
```

Formats can also be specified using a string, either in the fxp `dtype` format,
or by using `Qm.n` or `UQm.n` notation (or the equivalent `Sm.n`/`Um.n` notation).

```python
x = Fxp(-7.25, dtype='fxp-s16/8')
x = Fxp(-7.25, dtype='S8.8')
```

You can print more information only changing the verbosity of *info* method.

```python
x.info(verbose=3)
```

> dtype           =       fxp-s16/8  
> Value           =       -7.25  
>  
> Signed          =       True  
> Word bits       =       16  
> Fract bits      =       8  
> Int bits        =       7  
> Val data type   =       `<class 'float'>`
>  
> Upper           =       127.99609375  
> Lower           =       -128.0  
> Precision       =       0.00390625  
> Overflow        =       saturate  
> Rounding        =       trunc  
> Shifting        =       expand  

### Representations

We can representate the value stored en `x` in several ways:

```python
x
```

> fxp-s16/8(-7.25)  

```python
x.get_val()     # return a Numpy array with the val/values in original data type representation
x()             # equivalent to x.get_val() or x.astype(self.vdtype)
```

> -7.25  

In different bases:

```python
x.bin()
x.bin(frac_dot=True)    # binary with fractional dot
x.base_repr(2)          # binary with sign symbol (not complement)
x.hex()
x.base_repr(16)         # hex with sign symbol (not complement)
```

> '1111100011000000'  
> '11111000.11000000'  
> '-11101000000'  
> '0xF8C0'  
> '-740'  

In different types:

```python
x.astype(int)
x.astype(float)
x.astype(complex)
```

> -8  
> -7.25  
> (-7.25+0j)  

**Note** that if we do:

```python
x.val
```

we will get the fixed point value stored in memory, like an integer value. Don't use this value for calculations, but in some cases you may need it.

### changing values

We can change the value of the variable in several ways:

```python
x(10.75)            # the simpliest way
x.set_val(2.125)    # another option
```

**DO NOT DO THIS**:

```python
x = 10.75           # wrong
```

because you are just modifying `x` type... it isn't a *Fxp* anymore, just a simple *float* right now.

The same as `x.val` gives you the raw underlying value, you can set that value with

```python
x.set_val(43, raw=True)
```

### changing size

If we want to resize our fxp variable we can do:

```python
x.resize(True, 8, 6)    # signed=True, n_word=8, n_frac=6
```

### data types supported

Fxp can handle following input data types:

* int, uint
* float
* complex
* list
* ndarrays (n-dimensional numpy arrays)
* strings (bin, hex, dec)
* Fxp objects

Here some examples:

```python
x(2)
x(-1.75)
x(-2.5 + 1j*0.25)
x([1.0, 1.5, 2.0])
x(np.random.uniform(size=(2,4)))
x('3.5')
x('0b11001010')
x('0xA4')
```

### indexing

If we had been save a list or array, we can use indexing just like:

```python
x[2] = 1.0      # modify the value in the index 2
print(x[2])
```

---

## arithmetic

*Fxp* supports some basic math operations like:

```python
0.75 + x    # add a constant
x - 0.125   # substract a constant
3 * x       # multiply by a constant
x / 1.5     # division by a constant
x // 1.5    # floor division by a constant
x % 2       # modulo
x ** 3      # power
```

This math operations using a Fxp and a constant returns a **new Fxp** object with a precision that depends of configuration of Fxp object, `x.config` for examples above.

**The constant is converted into a new Fxp object before math operation**, where the Fxp size for the constant operand is defined by `x.config.op_input_size` in examples above. The default value for `op_input_size` is 'best' (best enoguh precision to represent the constant value), but it could be used 'same' to force the constant's size equals to Fxp object size (x in the examples).

> **Important note**: 
> 
> A **power** operation with a constant could be time consuming if _base number_ has many fractional bits. It's recommended to do ```x ** Fxp(3)``` in the previous example.

The result of math operation is returned as a new Fxp object with a precision defined according to `x.config.const_op_sizing`. This parameter could be configured with following options: 'optimal', 'same' (default), 'fit', 'largest', 'smallest'. For math operations with constants, by default (`config.const_op_sizing = 'same'`), a Fxp with same size is returned.

In all these cases we can assign the result to a (Fxp) variable, or to the same (overwritting the old Fxp object).

```python
y = 3.25 * (x - 0.5)    # y is a new Fxp object
```

Math operations using **two or more Fxp** variables is also supported, returning a new Fxp object like before cases. The size of returned Fxp object depends of both Fxp operand's sizes and the `config.op_sizing` parameter of the first (left) Fxp object. By default, `config.op_sizing = 'optimal'`, so, the returned size depends also of the math operation type. For example, in the addition case, the integer size of returned Fxp is 1 bit larger than largest integer size of operands, and size of fractional part of returned Fxp is equal to largest fractional size of operands.

```python
# Fxp as operands
x1 = Fxp(-7.25, signed=True, n_word=16, n_frac=8)
x2 = Fxp(1.5, signed=True, n_word=16, n_frac=8)
x3 = Fxp(-0.5, signed=True, n_word=8, n_frac=7)

y = 2*x1 + x2 - 0.5     # y is a new Fxp object

y = x1*x3 - 3*x2        # y is a new Fxp object, again
```

If we need to model that the result of a math operation is stored in other fractional fixed-point variable with a particular format we should do the following:

```python
# variable to store a result
y = Fxp(None, signed=True, n_word=32, n_frac=16)

y.equal(x1*x3 - 3*x2)
```

At the end, we also have the possibility of get the value of a math operation and set that val in the varible created to store the result.

```python
y.set_val( (x1*x3 - 3*x2).get_val() )   # equivalent to y.equal(x1*x3 - 3*x2), but less elegant
y( (x1*x3 - 3*x2)() )                   # just a little more elegant
```

Another example could be a sin wave function represented in Fxp:

```python
import numpy as np

f = 5.0         # signal frequency
fs = 400.0      # sampling frequency
N = 1000        # number of samples

n = Fxp( list(range(N)) )                       # sample indices
y( 0.5 * np.sin(2 * np.pi * f * n() / fs) )     # a sin wave with 5.0 Hz of frequecy sampled at 400 samples per second
```

## logical (bitwise) operators

*Fxp* supports logical (bitwise) operations like *not* (*inverse*), *and*, *or*, *xor* with constants or others Fxp variables. It also supports bits *shifting* to the right and left.

```python
x & 0b1100110011110000          # Fxp var AND constant
x & Fxp('0b11001100.11110000')  # Fxp var AND other Fxp with same constant
x & y                           # x AND y, both previoulsy defined

~x      # bits inversion
x | y   # x OR y
x ^ y   # x XOR y

x << 5  # x shifted 5 bits to the left
x >> 3  # x shifted 3 bits to the right (filled with sign bit)
```

When logical operations are performed with a constant, this constant is converted to a Fxp with de same characteristics of Fxp operand.

## Comparisons

*Fxp* supoorts comparison operators with constants, other variables, or another Fxp.

```python
x > 5
x == y
# ... and other comparison availables
```

---

## behaviors

Fxp has embedded some behaviors to process the value to store.

### overflow / underflow

A Fxp has upper and lower limits to representate a fixed point value, those limits are define by fractional format (bit sizes). When we want to store a value that is outside those limits, Fxp has an **overflow** y process the value depending the behavior configured for this situation. The options are:

* *saturate* (default): the stored value is clipped to *upper* o *lower* level, as appropiate. For example, if upper limit is 15.75 and I'd want to store 18.00, the stored value will be 15.75.
* *wrap* : the stored value is wrapped inside valid range. For example, if we have a `fxp-s7/2` the lower limit is -16.00 and the upper +15.75, and I'd want to store 18.00, the stored value will be -14.00 (18.00 is 2.00 above upper limit, so is stored 2.00 above lower limit).

We can change this behavior doing:

```python
# at instantiation
x = Fxp(3.25, True, 16, 8, overflow='saturate')

# afer ...
x.overflow = 'saturate'
# or
x.overflow = 'wrap'
```

If we need to know which are the *upper* and *lower* limits, Fxp have those stored inside:

```python
print(x.upper)
print(x.lower)
```

It is important to know the Fxp doesn't raise a warning if *overflow* or *underflow* happens. The way to know that is checking field `status['overflow']` and `status['underflow']` of each Fxp.

### rounding

Until now we had been storing values in our Fxp that were represented without loss of precision, and that was because we defined enough amount of bit for word and fractional part. In other words, if we want to save the value -7.25, we need 1 bit for sign, at least 3 bits for integer (2^**3** = 8), and at least 2 bits for fractional (2^-**2** = 0.25). In this case our Fxp would have `fxp-s6/2` format.

But, if we want to change the value of our Fxp to -7.3, the precision is not enough and Fxp will store -7.25 again. That is because Fxp is **rounding** the value before storing as a fractional fixed point value. Fxp allows different types of rounding methods:

* *trunc* (default): The truncated value of the scalar (let's say `x`) will be the nearest fractional supported value which is closer to zero than `x` is. In short, the fractional part of the signed number `x` that is not supported, is discarded. Round to nearest fractional supported value towards zero.
* *around* : Evenly round of the given value to the nearest fractional supported value, for example: 1.5 is rounded to 2.0.
* *floor* : The floor of the scalar `x` is the largest fractional supported value `i`, such that i <= x. It is often denoted as $\lfloor x \rfloor$.
* *ceil* :  The ceil of the scalar `x` is the smallest fractional supported value `i`, such that i >= x. It is often denoted as \lceil x \rceil.
* *fix* : Round to nearest fractional supported value towards zero.

We can change this behavior doing:

```python
# at instantiation
x = Fxp(3.25, True, 16, 8, rounding='floor')

# after ...
x.rounding = 'trunc'
# or ...
x.rounding = 'around'
x.rounding = 'floor'
x.rounding = 'ceil'
x.rounding = 'fix'
```

If we want to know what is the **precision** of our Fxp, we can do:

```python
print(x.precision)              # print the precision of x

# or, in a generic way:
print(Fxp(n_frac=7).precision)  # print the precision of a fxp with 7 bits for fractional part.
```

### inaccuracy

When the input value couldn't be represented exactly as a fixed-point, a **inaccuracy** flag is raised in the status of Fxp variable. You can check this flag to know if you are carrying a precision error.

## Status flags

*Fxp* have **status flags** to show that some events have occured inside the variable. The status flags are:

* overflow
* underflow
* inaccuracy

Those can be checked using:

```python
x.get_status()  # returns a dictionary with the flags

# or
x.get_status(format=str)    # return a string with flags RAISED only
```

The method **reset** can be call to reset status flags raised.

```python
x.reset()
```

---

## copy

We can copy a Fxp just like:

```python
y = x.copy()        # copy also the value stored
# or
y = x.deepcopy()

# if you want to preserve a value previously stored in `y` and only copy the properties from `x`:
y = y.like(x)
```

This prevent to redefine once and once again a Fxp object with same properties. If we want to modify the value en same line, we can do:

```python
y = x.copy()(-1.25)     # where -1.25 y the new value for `y` after copying `x`. It isn't necessary the `y` exists previously.
# or
y = Fxp(-1.25).like(x)
# or
y = Fxp(-1.25, like=x)

# be careful with:
y = y(-1.25).like(x)    # value -1.25 could be modify by overflow or rounding before considerating `x` properties.
y = y.like(x)(-1.25)    # better!
```

It is a good idea create Fxp objects like **template**:

```python
# Fxp like templates
DATA        = Fxp(None, True, 24, 15)
ADDERS      = Fxp(None, True, 40, 16)
MULTIPLIERS = Fxp(None, True, 24, 8)
CONSTANTS   = Fxp(None, True, 8, 4)

# init
x1 = Fxp(-3.2).like(DATA)
x2 = Fxp(25.5).like(DATA)
c  = Fxp(2.65).like(CONSTANTS)
m  = Fxp().like(MULTIPLIERS)
y  = Fxp().like(ADDERS)

# do the calc!
m.equal(c*x2)
y.equal(x1 + m)

```

## Scaling

*Fxp* implements an alternative way to input data and represent it, as an linear transformation through *scale* and *bias*. In this way, the raw fracitonal value stored in Fxp variable is "scaled down" during input and "scaled up" during output or operations.

It allows to use less bits to represent numbers in a huge range and/or offset.

For example, suppose that the set of numbers to represent are in [10000, 12000] range, and the precision needed is 0.5. We have 4000 numbers to represent, at least. Using scaling we can avoid to represent 12000 number or more. So, we only need 12 bits (4096) values.

```python
x = Fxp(10128.5, signed=False, n_word=12, scale=1, bias=10000)

x.info(3)
```

> dtype           =       fxp-u12/1  
> Value           =       10128.5  
> Scaling         =       1 * val + 10000  
>  
> Signed          =       False  
> Word bits       =       12  
> Fract bits      =       1  
> Int bits        =       11  
> Val data type   =       `<class 'float'>`
>  
> Upper           =       12047.5  
> Lower           =       10000.0  
> Precision       =       0.5  
> Overflow        =       saturate  
> Rounding        =       trunc  
> Shifting        =       expand  

Note that *upper* and *lower* limits are correct, and that the *precision* is what we needed.
