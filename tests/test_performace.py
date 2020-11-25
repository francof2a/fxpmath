import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import fxpmath as fxp
from fxpmath.objects import Fxp
from fxpmath import utils

import numpy as np
import time

def test_perf_clip(repeat=10):
    exec_time_vals = np.zeros(repeat)
    for i in range(repeat):
        start_time = time.time()
        utils.clip(np.random.uniform(low=-10, high=10, size=[1000,2]), val_min=-5, val_max=5 )
        exec_time_vals[i] = time.time() - start_time
    print('\nutils.clip execution time over {} repetitions'.format(repeat))
    print('\tmean = {:.3f} ms\n\tstd = {:.3f} ms'.format(np.mean(exec_time_vals)*1e3, np.std(exec_time_vals)*1e3))

    exec_time_vals = np.zeros(repeat)
    for i in range(repeat):
        start_time = time.time()
        np.clip(np.random.uniform(low=-10, high=10, size=[1000,2]), a_min=-5, a_max=5 )
        exec_time_vals[i] = time.time() - start_time
    print('\nnumpy.clip execution time over {} repetitions'.format(repeat))
    print('\tmean = {:.3f} ms\n\tstd = {:.3f} ms'.format(np.mean(exec_time_vals)*1e3, np.std(exec_time_vals)*1e3))


def test_perf_wrap(signed=True, n_word=8, repeat=10):
    exec_time_vals = np.zeros(repeat)
    for i in range(repeat):
        start_time = time.time()
        utils.wrap(np.random.uniform(low=-512, high=512, size=[1000,2]), signed=signed, n_word=n_word)
        exec_time_vals[i] = time.time() - start_time
    print('\nutils.wrap execution time over {} repetitions'.format(repeat))
    print('\tmean = {:.3f} ms\n\tstd = {:.3f} ms'.format(np.mean(exec_time_vals)*1e3, np.std(exec_time_vals)*1e3))


def test_perf_change_2d_indexed_value(m=24, n=16, repeat=10):

    A_fxp = Fxp(np.random.uniform(size=[m,n]))
    x = 1/repeat

    exec_time_vals = np.zeros(repeat)
    for r in range(repeat):
        start_time = time.time()
        for i in range(m):
            for j in range(n):
                    A_fxp[i,j] = A_fxp[i,j] + x
        exec_time_vals[r] = time.time() - start_time
    print('\ntest_perf_change_2d_indexed_value execution time over {} repetitions'.format(repeat))
    print('\tmean = {:.3f} ms\n\tstd = {:.3f} ms'.format(np.mean(exec_time_vals)*1e3, np.std(exec_time_vals)*1e3))

if __name__ == "__main__":
    test_perf_clip()

    test_perf_wrap()

    test_perf_change_2d_indexed_value()