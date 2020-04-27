# fxpmath

A python library for fractional fixed-point arithmetic.

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
> Signed          =       True  
> Word bits       =       6  
> Fract bits      =       2  
> Int bits        =       3  

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

### Representations

We can representate the value stored en `x` in several ways:

```python
x
```

> -7.25

```python
x.get_val()     # return the val with original type
x()             # equivalent to x.get_val() or x.astype(self.vdtype)
```

> -7.25

In different bases:

```python
x.bin()
x.hex()
x.base_repr(2)  # binary with sign symbol (not complement)
x.base_repr(16) # hex with sign symbol (not complement)
```

> '1111100011000000'  
> '0xf8c0'  
> '-11101000000'  
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

... yes **fxpmath** supports *complex* numbers!

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

Here some examples:

```python
x(2)
x(-1.75)
x(-2.5 + 1j*0.25)
x([1.0, 1.5, 2.0])
x(np.random.uniform(size=(2,4)))
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
```

This math operations returns a **new Fxp** object with automatic precision enough to represent the result. So, in all these cases we can assign the result to a (Fxp) variable, or to the same (overwritting the old Fxp object).

```python
y = 3.25 * (x - 0.5)    # y is a new Fxp object
```

Math operations using **two or more Fxp** variables is also supported, returning a new Fxp object like before cases. If we add two Fxp, the numbers of bits of resulting Fxp is a 1 more bit; if we multiply two Fxp, the numbers of bits of resulting Fxp is a the sum of the bits of operands. In case of division is different, because the resulting number of bits is calculated to get the enough precision to represent the result.

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

* *trunc* (default): The truncated value of the scalar (let's say `x`) will be the nearest fractional supported value which is closer to zero than `x` is. In short, the fractional part of the signed number `x` that is not supported, is discarded.
* *around* : Evenly round of the given value to the nearest fractional supported value.
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
