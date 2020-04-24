# fxpmath

A python library for fractional fixed-point arithmetic.

## install

coming soon!

## quick start

### creation

Let's jump into create our new **fractional fixed-point** variable:

```python
from fxpmath import Fxp

x = Fxp(-7.25) # create fxp variable with value 7.25
x.info()
```

>        dtype           =       fxp-s6/2
>        Value           =       -7.25
>        Signed          =       True
>        Word bits       =       6
>        Fract bits      =       2
>        Int bits        =       3

We have created a variable of 6 bits, where 1 bit has been reserved for sign, 2 bits for fractional part, and 3 remains for integer part. Here, bit sizes had been calculated to just satisfy the value you want to save.

But the most common way to create a **fxp** variable beside the its value is defining explicitly if it is signed, the number of bits for whole word and for the fractional part.

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
x()
```

> -7.25

In different bases:

```python
x.bin()
```

> '100011'

```python
x.hex()
```

> '0x23'

In different types:

```python
x.astype(int)
```

> -8

```python
x.astype(float)
```

> -7.25

```python
x.astype(complex)
```

> (-7.25+0j)

... yes **fxpmath** supports *complex* numbers!

### changing values and sizes

We can change the value of the variable in several ways:

```python
x(10.75) # the simpliest way
x.set_val(2.125) # another option
```

Note that if we do:

```python
x.val
```

we will get the fixed point value stored in memory, like an integer value. Don't use this value for calculations, but in some cases you may need it.

If we want to resize our fxp variable we can do:

```python
x.resize(True, 8, 6) # signed=True, n_word=8, n_frac=6
```
