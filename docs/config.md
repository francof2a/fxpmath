# Config

The class `config` is a parameters set which determines the behavior of a Fxp object in several processes.

Let's take a look over this class:

```python
x = Fxp(None, True, 16, 4)
x.config.print()
```

> max_error               :       1.0842021724855044e-19  
> n_word_max              :       64  
> overflow                :       saturate  
> rounding                :       trunc  
> shifting                :       expand  
> op_method               :       raw  
> op_input_size           :       same  
> op_out                  :       None  
> op_out_like             :       None  
> op_sizing               :       optimal  
> const_op_sizing         :       same  
> array_output_type       :       fxp  
> array_op_out            :       None  
> array_op_out_like       :       None  
> array_op_method         :       repr  
> dtype_notation          :       fxp  
> bin_prefix              :       None  
> hex_prefix              :       0x

The first 5 parameters were present in older versions, but they were move here. Note that size parameters don't live here.

Following sections will explain breafly rest of parameters.

# op_method

This parameter defines if a operation involving its Fxp object is going to be performed using `raw` or `repr` method.

The `raw` method use the `int` internal representation of data, whilst `repr` use the original type of data for the operation. The `raw` method is more accurate and emulate binary level operation, but is slower most of cases. This method is needed in cases of *extended precision*.

```python
a = Fxp([-1.0, 0.0, 1.0], True, 128, 96)
b = Fxp([2**-64, -2**48, 2**32], True, 128, 96)

c = a + b
print(c)
print(c-a) # equal to b
```

> [-1.0 -2147483648.0 2147483649.0]  
> [5.421010862427522e-20 -2147483648.0 2147483648.0]  

```python
a.config.op_method = 'repr'
b.config.op_method = 'repr'

c = a + b
print(c)
print(c-a) # should be equal to b, but it doesn't
```

> [-1.0 -2147483648.0 2147483649.0]  
> [0.0 -2147483648.0 2147483648.0]  

Using `repr`, addition is equivalent to $c = \text{Fxp}(a() + b())$. The fist value of $c - a$ is not equal to first value of $b$ because $2^{-64}$ has been lost when it was added to $1.0$ becuase *float precision* limit.

## op_input_size

When an arithmetic operator is used between an Fxp and a non-Fxp object, last is going to be converted to Fxp before operation is performed. Sizing process of this non-Fxp object is specified by `op_input_size`. 

The values could be `'same'` (default, same size than Fxp) or `'best'` (best size according value(s) of non-Fxp).

```python
x = Fxp(2.0, True, 16, 2)
x.config.op_input_size = 'same'

z = x * 2.125

z.info()
```
  
> dtype		=	fxp-s16/2  
> Value		=	4.0  

Note that 2.125 is converted to Fxp(2.125, True, 16, 2), losing precision.

```python
x = Fxp(2.0, True, 16, 2)
x.config.op_input_size = 'best'

z = x * 2.125

z.info()
```

> dtype		=	fxp-s16/2  
> Value		=	4.25  

Now, the value 2.125 is converted to Fxp using the best resolution in order to represent the real value before the aritmetic operation.

## op_out

This can point to a specific Fxp object to store a result. By default this is `None`.

If two Fxp objects are used the first will be the reference.

```python
x = Fxp(2.0, True, 16, 2)
z = Fxp(0.0, True, 16, 4)

x.config.op_out = z

(x + 1).info()
z.info()
```

> dtype		=	fxp-s16/4  
> Value		=	3.0  
> 
> dtype		=	fxp-s16/4  
> Value		=	3.0  

## op_out_like

This defines the Fxp type for result. By default this is `None`.

If two Fxp objects are used the first will be the reference.

```python
x = Fxp(2.0, True, 16, 2)
x.config.op_out = Fxp(None, True, 24, 4)

z = x * 3.5 

z.info()
```

> dtype		=	fxp-s24/4  
> Value		=	7.0  

## op_sizing

This defines size of Fxp returned by an operation, applied to one or two Fxp objects. The option for config this parameters are: 'optimal', 'same', 'fit', 'largest', 'smallest'. Where:

- `'optimal'`: size is theorical optimal according operation and sizes of operands.
- `'same'`: same size that firs operand.
- `'fit'`: best size is calculated according result value(s).
- `'largest'`: same size that largest (size) Fxp operand.
- `'smallest'`: same size that smallest (size) Fxp operand.

```python
x = Fxp(3.5, True, 16, 4)
y = Fxp(-1.25, True, 24, 8)

for s in x.config._op_sizing_list:
    x.config.op_sizing = s
    print('{}:'.format(x.config.op_sizing))
    (x + y).info()
```
<pre>
optimal:  
    dtype		=	fxp-s25/8  
    Value		=	2.25  

same:  
    dtype		=	fxp-s16/4  
    Value		=	2.25  

fit:  
    dtype		=	fxp-s11/8  
    Value		=	2.25  

largest:  
    dtype		=	fxp-s24/8  
    Value		=	2.25  

smallest:  
    dtype		=	fxp-s16/4  
    Value		=	2.25  
</pre>

## const_op_sizing

This defines the same behavior that `op_sizing` parameter, but when a constant is used as operand instead other Fxp object. First, constat is converted to Fxp according to `op_input_size` and then output Fxp (result) size is choosen by `const_op_sizing`.

```python
x = Fxp(3.5, True, 16, 4)

for s in x.config._const_op_sizing_list:
    x.config.const_op_sizing = s
    print('{}:'.format(x.config.const_op_sizing))
    (x + 2).info()
```
<pre>
optimal:
	dtype		=	fxp-s17/4
	Value		=	5.5

same:
	dtype		=	fxp-s16/4
	Value		=	5.5

fit:
	dtype		=	fxp-s8/4
	Value		=	5.5

largest:
	dtype		=	fxp-s16/4
	Value		=	5.5

smallest:
	dtype		=	fxp-s16/4
	Value		=	5.5
</pre>

## array_output_type

This defines the type of the object get as result when use array (numpy) functions. The options are: 'fxp' (defautl, return an Fxp) or 'array' (return an numpy n-dimensional array).

```python
x = Fxp([1.0, 2.5, 3.0, 4.5], dtype='fxp-s16/4')
x.config.array_output_type = 'fxp'

z = np.sum(x)
z.info()

x.config.array_output_type = 'array'
z = np.sum(x)
print(z, type(z))
```
<pre>
	dtype		=	fxp-s18/4
	Value		=	11.0

11.0 &#60;class 'numpy.ndarray'&#62;
</pre>

## array_op_out

Same as `op_output` but used with numpy functions:

```python
w = Fxp([1.0, -2.5, 3.0, -4.5], dtype='fxp-s16/4')
z = Fxp(0.0, True, 16, 4)
w.config.array_op_out = z

y = np.sum(w)
y.info()
z.info()
print(y is z)
```
<pre>
	dtype		=	fxp-s16/4
	Value		=	-3.0

	dtype		=	fxp-s16/4
	Value		=	-3.0

True
</pre>

## array_op_out_like

Same as `op_out_like` but used with numpy functions:

```python
w = Fxp([1.0, -2.5, 3.0, -4.5], dtype='fxp-s16/4')
w.config.array_op_out_like = Fxp(0.0, True, 32, 8)

y = np.sum(w)
y.info()

y = np.cumsum(w)
y.info()
```
<pre>
	dtype		=	fxp-s32/8
	Value		=	-3.0

	dtype		=	fxp-s32/8
	Value		=	[ 1.  -1.5  1.5 -3. ]
</pre>

## array_op_method

This property defines what values are used for (numpy) arrays created from Fxp objects. If 'raw', arrays have *raw values* of Fxp object, whilst 'repr', arrays have values in original (representation) type.

This parameter is used also to defined kernel op_method when array conversion is involved.

```python
w = Fxp([1.0, -2.5, 3.0, -4.5], dtype='fxp-s8/2')
w.config.array_op_method = 'repr'

w_array = np.asarray(w)
print(w_array, type(w_array))

w.config.array_op_method = 'raw'
w_array = np.asarray(w)
print(w_array, type(w_array))
```
<pre>
[ 1.  -2.5  3.  -4.5] &#60;class 'numpy.ndarray'&#62;
[  4 -10  12 -18] &#60;class 'numpy.ndarray'&#62;
</pre>
