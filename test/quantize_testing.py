#%%
import sys
import numpy as np 
import matplotlib.pyplot as plt 

sys.path.append('..')
import fxpmath as fxp
from fxpmath.objects import Fxp

#%%
fs = 1.0e6
f = 1.0e3
T = 2e-3
Nsim = np.ceil(T*fs).astype(int)
n = np.linspace(0, Nsim-1, Nsim)
t = n/fs
A = 5.0

signal = A * np.sin(2*np.pi*f*t)

# plt.figure()
# plt.plot(t, signal)
# plt.show()

# %%
signal_q = Fxp(signal, n_word=8, n_frac=1)

plt.figure(figsize=(12,8))
plt.plot(t, signal, color='C0')
plt.plot(t, signal_q.astype(float), color='C1')
plt.show()
# %%
plt.figure(figsize=(12,10))
plt.plot(t, signal, color='C0')

roungings = ['floor', 'ceil', 'around', 'fix', 'trunc']
signals = []
for r in roungings:
    signal_q = Fxp(n_word=8, n_frac=1)
    signal_q.props['rounding'] = r
    signal_q.set_val(signal)
    signals.append(signal_q)

    plt.plot(t, signal_q.astype(float), label=r)
plt.legend()
plt.show()

# %%
