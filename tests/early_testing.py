#%%
import sys
import numpy as np

sys.path.append('..')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import fxpmath as fxp
from fxpmath.objects import Fxp
from fxpmath.functions import *

#%%
x = Fxp()
print(x, x.signed, x.n_word, x.n_frac)

x = Fxp(4.125)
print(x, x.signed, x.n_word, x.n_frac)

x = Fxp(-4.125)
print(x, x.signed, x.n_word, x.n_frac)

x = Fxp(4.126)
print(x, x.signed, x.n_word, x.n_frac)

x = Fxp(4.126, n_frac=3)
print(x, x.signed, x.n_word, x.n_frac)

x = Fxp(7.75)
print(x, x.signed, x.n_word, x.n_frac)

x = Fxp(7.75, n_word=16)
print(x, x.signed, x.n_word, x.n_frac)

x = Fxp([-4.0, 0.001, 3.75, 31.0])
print(x, x.signed, x.n_word, x.n_frac)

x = Fxp([-4.0, 0.002, 3.75, 31.0], max_error=0.001)
print(x, x.signed, x.n_word, x.n_frac)

#%%
x = Fxp(4.25, signed=True, n_word=8, n_frac=2)
print(x.val)
print(x)

# %%
x = Fxp([-33.0, -32.0, -31.0, -0.25, 0.0, 0.25, 0.5, 31.0, 32.0, 33.0], signed=True, n_word=8, n_frac=2)
print(x.val)
print(x)

# %%
x1 = Fxp(4.25, signed=True, n_word=8, n_frac=2)
x2 = Fxp(1.0, signed=True, n_word=8, n_frac=2)

y = x1 + x2
print('{} + {} = {}'.format(x1, x2, y))

x2 = 20.5
y = x1 + x2
print('{} + {} = {}'.format(x1, x2, y))

x2 = 1.0
y = x1 - x2
print('{} - {} = {}'.format(x1, x2, y))

#%%
x1 = Fxp(4.25, signed=True, n_word=8, n_frac=4)
x2 = Fxp(1.0, signed=True, n_word=16, n_frac=2)

y = x1 + x2
print('{} + {} = {}'.format(x1, x2, y))
print(y.info())

#%%
y = Fxp(signed=True, n_word=8, n_frac=4)

x1 = Fxp(4.25, signed=True, n_word=8, n_frac=4)
x2 = Fxp(1.0, signed=True, n_word=16, n_frac=2)

y.equal(x1 + x2)
print('{} + {} = {}'.format(x1, x2, y))
print(y.info())

x3 = Fxp(15.0, signed=True, n_word=16, n_frac=2)

y.equal(x1 + x3)
print('{} + {} = {}'.format(x1, x3, y))
print(y.info())

y.equal(x1 - x3)
print('{} - {} = {}'.format(x1, x3, y))
print(y.info())

#%%
x1 = Fxp(4.25, signed=True, n_word=8, n_frac=4)
c = 2.0
y = x1 * c
print('{} * {} = {}'.format(x1, c, y))
print(y.info())

c = -1.5
y = c * x1
print('{} * {} = {}'.format(c, x1, y))
print(y.info())

x2 = Fxp(-1.0, signed=True, n_word=16, n_frac=2)

y = x1 * x2
print('{} * {} = {}'.format(x1, x2, y))
print(y.info())

x3 = Fxp([-1.0, 1.0, 0.0, 10.5], signed=True, n_word=16, n_frac=2)

y = x1 * x3
print('{} * {} = {}'.format(x1, x3, y))
print(y.info())

#%%
print('\n--- fxp_like ---')
x_ref = Fxp(signed=True, n_word=8, n_frac=4)

x1 = fxp_like(x_ref, val=2.1)
x2 = fxp_like(x_ref, val=-0.25)
y = fxp_like(x_ref)

y.equal(x1 + x2)
print('{} + {} = {}'.format(x1, x2, y))
print(y.info())


#%%
print('\n--- fxp_like ---')
x_ref = Fxp(signed=True, n_word=8, n_frac=4)

x1 = fxp_like(x_ref, val=[0.125, 0.25, 0.5, 0.625, 0.75])

y = x1 + x1
print('{} + {} = {}'.format(x1, x1, y))


#%%
print('\n--- COMPLEX NUMBERS ---')
x = Fxp(0.125 + 1j*0.25, signed=True, n_word=16, n_frac=14)
print(x.info())
print(x)
print(x.astype(complex))
print(x.real)
print(x.imag)

x = Fxp([0.125 + 1j*0.25, 1.0 - 1j*1.0], signed=True, n_word=16, n_frac=14)
print(x.info())
print(x)
print(x.real)
print(x.imag)

x = Fxp([0.125 + 1j*0.25, 1.0 - 1j*1.0], signed=True)
print(x.info())

#%%
print('\n--- WRAP OVERFLOW ---')
x = Fxp(signed=True, n_word=6, n_frac=2)
x.props['overflow'] = 'wrap'
x.set_val(8)
print(x)
x.set_val(np.arange(-10,10,1))
print(x)
x.props['overflow'] = 'saturate'
x.set_val(np.arange(-10,10,1))
print(x)

x = Fxp(signed=False, n_word=6, n_frac=2)
x.props['overflow'] = 'wrap'
x.set_val(np.arange(-2,18,1))
print(x)
x.props['overflow'] = 'saturate'
x.set_val(np.arange(-2,18,1))
print(x)

#%%
print('\n--- ROUNDING ---')
print('\n* around *')
x = Fxp(signed=True, n_word=8, n_frac=2)
x.props['rounding'] = 'around'
values = np.arange(30.0, 32.5, 0.125)
x.set_val(values)
print(values)
print(x)

print('\n* floor *')
x = Fxp(signed=True, n_word=8, n_frac=2)
x.props['rounding'] = 'floor'
values = np.arange(30.0, 32.5, 0.125)
x.set_val(values)
print(values)
print(x)

print('\n* ceil *')
x = Fxp(signed=True, n_word=8, n_frac=2)
x.props['rounding'] = 'ceil'
values = np.arange(30.0, 32.5, 0.125)
x.set_val(values)
print(values)
print(x)

print('\n* fix *')
x = Fxp(signed=True, n_word=8, n_frac=2)
x.props['rounding'] = 'fix'
values = np.arange(30.0, 32.5, 0.125)
x.set_val(values)
print(values)
print(x)

print('\n* trunc *')
x = Fxp(signed=True, n_word=8, n_frac=2)
x.props['rounding'] = 'trunc'
values = np.arange(30.0, 32.5, 0.125)
x.set_val(values)
print(values)
print(x)