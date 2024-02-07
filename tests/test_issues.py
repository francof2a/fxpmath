import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import fxpmath as fxp
from fxpmath.objects import Fxp, Config
from fxpmath import functions

import numpy as np

def test_issue_9_v0_3_6():
    M = 24
    N = 16
    A_fxp = Fxp(np.zeros((M,N)), True, 16, 8)

    # indexed element returned as Fxp object
    assert isinstance(A_fxp[0], Fxp)

    # accept an Fxp object like input value
    x = Fxp(4, True, 8, 2)
    A_fxp[0,0] = x
    assert A_fxp[0,0]() == 4

    B_fxp = Fxp(x, like=A_fxp)
    assert B_fxp() == 4

    C_fxp = Fxp(x)
    assert C_fxp() == 4

def test_issue_10_v0_3_6():
    x = Fxp(1.5, True, 256, 64)
    assert x() == 1.5

def test_issue_11_v0_3_6():
    try:
        val = np.float128(1.5)
    except:
        val = None
    
    if val is not None:
        x = Fxp(val, True, 256, 64)
        assert x() == 1.5

    # x = Fxp(np.float128(1.5), True, 256, 64)
    # assert x() == 1.5

def test_issue_14_v0_3_7():
    # d = Fxp('0b000.00000000000000000001011101010110101011101010101010101010101010101010101010101001010101010101010101001010101001010101010101010', 
    #         True, n_word=128, n_frac=125, rounding='around')
    # assert d.bin(frac_dot=True) == '000.00000000000000000001011101010110101011101010101010101010101010101010101010101001010101010101010101001010101001010101010101010'

    d = Fxp('0b00000000000000000000001011101010110101011101010101010101010101010101010101010101001010101010101010101001010101001010101010101010', 
            True, n_word=128, n_frac=125, rounding='around', raw=True)
    assert d.bin() == '00000000000000000000001011101010110101011101010101010101010101010101010101010101001010101010101010101001010101001010101010101010'
 
def test_issue_15_v0_3_7():
    x = Fxp('0xb', True, 10, 4)
    assert x.hex() == '0x00B'

def test_issue_17_v0_3_7():
    a = Fxp(15, signed=False)
    b = a ** Fxp(2)

    assert b() == 15**2

def test_issue_19_v0_3_7():
    DW=12
    DATA_FXPTYPE = Fxp(None, signed=True, n_word=DW, n_frac=DW-1)

    a = Fxp(np.zeros(2, dtype=complex), like=DATA_FXPTYPE)
    c = Fxp(complex(0,0), like=DATA_FXPTYPE)
    b = Fxp(0.5-0.125j, like=DATA_FXPTYPE)

    # print(c.get_val() + b.get_val())
    c.equal(c+b)
    
    # c.info()
    assert c() == 0.5-0.125j

    # print(a[0].get_val() + b.get_val())
    # a[0].equal(a[0]+b) # not supported
    a[0] = a[0]+b
    
    # a[0].info()
    assert a[0]() == 0.5-0.125j

def test_issue_20_v0_3_8():
    x = Fxp(0, signed=True, n_word = 4, n_frac = 0, overflow='wrap')

    assert x(-30) == 2
    assert x(-8) == -8
    assert x(-9) == 7
    assert x(7) == 7
    assert x(8) == -8

def test_issue_21_v0_3_8():
    a = [1, 2, 3]
    b = [0, 1, 0]
    assert (np.inner(a, b) == 2)

    na = np.array([1, 2, 3])
    nb = np.array([0, 1, 0])
    assert (np.inner(na, nb) == np.inner(a, b)).all()

    fa = Fxp([1, 2, 3])
    fb = Fxp([0, 1, 0])
    z = np.inner(fa, fb)
    assert (np.inner(fa, fb)() == np.inner(a, b)).all()

def test_issue_26_v0_4_0():
    sig = np.array(['0xff864d8f', '0xff86b76d', '0xff880f87'])

    fxp_sig = Fxp(sig)
    assert fxp_sig[0] == -7975537
    assert fxp_sig[1] == -7948435
    assert fxp_sig[2] == -7860345

    fxp_sig = Fxp(sig, signed=False)
    assert fxp_sig[0] == int('0xff864d8f', 16)
    assert fxp_sig[1] == int('0xff86b76d', 16)
    assert fxp_sig[2] == int('0xff880f87', 16)

def test_issue_31_v0_4_0():
    t = Fxp(2**32, dtype="u32.32", shifting="trunc")
    assert t.status['extended_prec'] == True
    assert t.val.dtype == object
    assert t.val == 2**64 - 1


    s = t(0.125)
    assert s.val.dtype == object
    assert t() == 0.125


    q = t(0.125)<<3
    assert q.val.dtype == object
    assert q() == 1.0


    q2 = t(0.125)*2**3
    assert q2.val.dtype == object
    assert q2() == 1.0


    q3 = t(0.125*2**3)
    assert q3.val.dtype == object
    assert q3() == 1.0

def test_issue_41_v0_4_2():
    x = Fxp(2, False, 63, 0, overflow='wrap')
    y = Fxp(2, False, 64, 0, overflow='wrap')

    assert x() == 2
    assert y() == 2

    x = Fxp(2, False, 31, 0, overflow='wrap')
    y = Fxp(2, False, 32, 0, overflow='wrap')

    assert x() == 2
    assert y() == 2

    x = Fxp(2.5, signed=True, n_word=31, n_frac=24, overflow='wrap')
    y = Fxp(2.5, signed=True, n_word=32, n_frac=24, overflow='wrap')

    assert x() == 2.5
    assert y() == 2.5

    x = Fxp(2.5, signed=True, n_word=63, n_frac=48, overflow='wrap')
    y = Fxp(2.5, signed=True, n_word=64, n_frac=48, overflow='wrap')

    assert x() == 2.5
    assert y() == 2.5


def test_issue_42_v0_4_2():
    b = Fxp(2, True, 4, 0, overflow='wrap')
    assert (b + 8)() == -6.0
    assert (b - 8)() == -6.0

def test_issue_44_v0_4_3():
    # 1a
    b = Fxp(20.5, False, n_word=5, scaling=1, bias=8)
    assert b() == 20.5

    # 1b
    b = Fxp(20.5, False, n_word=4, scaling=1, bias=8)
    assert b() == 20
    b = Fxp(20.5, False, n_word=4, n_frac=1, scaling=1, bias=8)
    assert b() == 15.5

    # 2
    zero = Fxp(0.0, False, n_word=5, overflow='wrap', scaling=1, bias=8)
    assert zero() == 32
    zero = Fxp(0.0, False, n_word=5, n_frac=1, overflow='wrap', scaling=1, bias=8)
    assert zero() == 16.0
    assert zero.upper == 23.5
    assert zero.lower == 8.0

    # 3
    b = Fxp(0, False, 64, 0, overflow='wrap', bias=8)
    assert b() == 2**64
    b = Fxp(8, False, 64, 0, overflow='wrap', bias=8)
    assert b() == 8
    b = Fxp(2**64 + 7, False, 64, 0, overflow='wrap', bias=8)
    assert b() == 2**64 + 7
    b = Fxp(2**64 + 8, False, 64, 0, overflow='wrap', bias=8)
    assert b() == 8

    b = Fxp(2**64+6, False, 64, 0, overflow='wrap', scaling=2, bias=8)
    assert b() == 2**64+6

def test_issue_48_v0_4_8():
    """
    Flags not propagated
    https://github.com/francof2a/fxpmath/issues/48
    """
    a = Fxp(-2., dtype="fxp-s24/8")
    b = Fxp(2.15, dtype="fxp-s24/8")
    assert b.status['inaccuracy']

    # inaccuracy in b must be propagated to c
    c = a + b
    assert c.status['inaccuracy']

    # add extra test using a inaccurate Fxp to set a new Fxp
    d = Fxp(c)
    assert d.status['inaccuracy']

def test_issue_49_v0_4_8():
    """
    Reversal of .bin()
    https://github.com/francof2a/fxpmath/issues/49
    """
    # Method 1
    x1 = Fxp(3.4)
    x_bin = x1.bin()
    x2 = Fxp('0b' + x_bin, like=x1)
    assert x1 == x2

    # Method 2
    x_bin = x1.bin(frac_dot=True)
    x2 = Fxp('0b' + x_bin)
    assert x1 == x2

    # Method 3
    x_bin = x1.bin()
    x2 = Fxp(like=x1).from_bin(x_bin)
    assert x1 == x2

    x_bin = x1.bin(frac_dot=True)
    x2 = Fxp(like=x1).from_bin(x_bin)
    assert x1 == x2

    # Method 4
    x_bin = x1.bin(frac_dot=True)
    x2 = functions.from_bin(x_bin)
    assert x1 == x2

    # alternatives to get binary string with prefix
    x_bin = x1.bin(frac_dot=True, prefix='0b')
    x2 = Fxp(x_bin)
    assert x1 == x2

    x1.config.bin_prefix = '0b'
    x_bin = x1.bin(frac_dot=True)
    x2 = Fxp(x_bin)
    assert x1 == x2

    # test negative value
    x1 = Fxp(-3.4)
    x_bin = x1.bin(frac_dot=True)
    x2 = functions.from_bin(x_bin)
    assert x1 == x2

    # test raw value
    x_bin = x1.bin()
    x2 = functions.from_bin(x_bin, raw=True, like=x1)
    assert x1 == x2

    # test complex value
    x1 = Fxp(-3.4 + 1j*0.25)
    x_bin = x1.bin(frac_dot=True)
    x2 = functions.from_bin(x_bin)
    assert x1 == x2


def test_issue_53_v0_4_5():
    x = Fxp(2j, dtype = 'fxp-u4/0-complex')
    z = x/2

    assert z() == 1j

def test_issue_55_v0_4_5():
    x = Fxp(0b11+0b11*1j, dtype = 'fxp-u2/0-complex')
    z = x & 0b01

def test_issue_56_v0_4_5():
    arr_fxp = Fxp(np.array([[1, 2]]))
    assert np.all(arr_fxp.bin() == np.array(['001', '010']))

def test_issue_58_v0_4_5():
    # datatype definition
    TAP = Fxp(None, dtype='fxp-s32/24-complex')
    SIGNAL = Fxp(None, dtype='fxp-s32/24-complex')

    # signal
    signal = np.array([(1 + 1j), (1 - 1j), (-1 + 1j), (-1 - 1j)], dtype=complex)
    signal_fxp = Fxp(signal).like(SIGNAL)
    signal_fxp1 = Fxp(signal, dtype='fxp-s32/24-complex')

    # generate filter
    filt = np.arange(0.1, 0.5, 0.1)
    filt_fxp = Fxp(filt).like(TAP)
    filt_fxp1 = Fxp(filt, dtype='fxp-s32/24')

    # convolve signal and filter
    out = np.convolve(signal_fxp, filt_fxp, 'same')
    out1 = np.convolve(signal_fxp1, filt_fxp1, 'same')

    assert np.all(signal_fxp() == signal_fxp1())
    assert np.all(filt_fxp() == filt_fxp1())
    assert np.all(out() == out1())

def test_issue_60_v0_4_6():
    cfg=Config(dtype_notation="Q",rounding="around")

    t_fxp = Fxp(0.0,1,n_int=16,n_frac=15,config=cfg)
    t_int = Fxp(0.0,1,n_int=13,n_frac=0,config=cfg)

    arr = [-5,0,14.8,7961.625]
    fullprec        = Fxp(arr      , like=t_fxp )
    rounded_direct  = Fxp(arr      , like=t_int )
    rounded         = Fxp(fullprec , like=t_int )
    assert np.all(rounded_direct == rounded)

    scalar_full         = Fxp(7961.625     , like=t_fxp )
    scalar_round_direct = Fxp(7961.625     , like=t_int )
    scalar_round        = Fxp(scalar_full  , like=t_int )

    assert scalar_round_direct == scalar_round

def test_issue_62_v0_4_7():
    y = Fxp(dtype='fxp-s6/2')
    y([[1.0,0.25,0.5],[0.25,0.5,0.25]])

    y[0][0] = -1.0

    assert y[0][0]() == -1.0

    y[0][0] = y[0][0] + 1.0

    assert y[0][0]() == 0.0

def test_issue_66_v0_4_8():
    x = Fxp(np.array([1.25, 0.5]), dtype='S8.4')
    y = Fxp(np.array([2.25, 1.5]), dtype='S16.6')
    # x[0].equal(y[0]) # it does NOT work
    # x[0] = y[0] # it works
    x.equal(y[0], index=0) # it works

    assert x[0]() == y[0]()

def test_issue_67_v0_4_8():
    input_size = Fxp(None, dtype='fxp-s32/23')
    f = [0,10+7j,20-0.65j,30]
    f = Fxp(f, like = input_size)

    def FFT(f):
        N = len(f)
        if N <= 1:
            return f

        # division: decompose N point FFT into N/2 point FFT
        even= FFT(f[0::2])
        odd = FFT(f[1::2])

        # store combination of results
        temp = np.zeros(N, dtype=complex)
        # temp = Fxp(temp, dtype='fxp-s65/23')
        temp = Fxp(temp, dtype='fxp-s65/46')

        for u in range(N//2):
            W =  Fxp(np.exp(-2j*np.pi*u/N), like=input_size) 
            temp[u] = even[u] + W* odd[u] 
            temp[u+N//2] = even[u] - W*odd[u]  
            
        return temp

    # testing the function to see if it matches the manual computation
    F_fft = FFT(f)
    
def test_issue_73_v0_4_8():
    # single unsigned value does work
    a = Fxp(10, False, 14, 3)
    b = Fxp(15, False, 14, 3)
    c = a - b
    assert c() == 0.0  # 0.0 --> correct

    # unsigned list does not work
    d = Fxp([10, 21], False, 14, 3)
    e = Fxp([15, 15], False, 14, 3)
    f = d - e
    assert f[0]() == 0.0  # [4095.875 6.0] --> 4095.875 is the upper limit
    assert f[1]() == 6.0

def test_issue_76_v0_4_8():
    # Numpy Issue with Bigger bit sizes
    # Getting strange results when using larger bit sizes in numpy calls
    
    # This works
    w = Fxp([1, 1, 1, 1], dtype='fxp-s29/0')
    y = np.cumsum(w)
    assert np.all(y() == np.array([1, 2, 3, 4]))

    # This doesn't
    w = Fxp([1, 1, 1, 1], dtype='fxp-s32/0')
    y = np.cumsum(w)
    assert np.all(y() == np.array([1, 2, 3, 4])) # works in linux, not in windows

    # Increase word size above 64 bits
    w = Fxp([1, 1, 1, 1], dtype='fxp-s64/0')
    y = np.cumsum(w)
    assert np.all(y() == np.array([1, 2, 3, 4]))

def test_issue_77_v0_4_8():
    # Precision error when numpy.reshape

    a = np.array([[0.762, 0.525], [0.345, 0.875]], dtype=complex)
    x = Fxp(a, signed=True, n_word=5, n_frac=3)
    # fxp-s5/3-complex
    assert x.signed == True and x.n_word == 5 and x.n_frac == 3

    y = np.reshape(x, (1, 4))
    # fxp-s4/3-complex
    assert y.signed == True and y.n_word == 5 and y.n_frac == 3

def test_issue_80_v0_4_8():
    # Creation of Fxp-object with negative n_frac

    # The following code results in unexpected behaviour 
    # when trying to specify the same type using alternative formats
    x = Fxp(16, signed=True, n_word=8, n_frac=-2)
    # -> x.dtype = 'fxp-s8/-2' , ok
    assert x.dtype == 'fxp-s8/-2'

    x = Fxp(16, dtype='S10.-2')
    assert x.dtype == 'fxp-s8/-2'

    x = Fxp(16, dtype='fxp-s8/-2')
    assert x.dtype == 'fxp-s8/-2'

def test_issue_85_v0_4_8():
    # Wrap overflow breaks on 0.0 value

    dt_values = ['fxp-s32/16', 'fxp-s64/32', 'fxp-s96/64']

    for dt in dt_values:
        x = Fxp(0,   dtype=dt)  # => Success
        assert x() == 0.0

        x = Fxp(0.0, dtype=dt)  # => Success
        assert x() == 0.0

        x = Fxp(0,   dtype=dt, overflow='wrap')  # => Success
        assert x() == 0.0

        x = Fxp(0.0, dtype=dt, overflow='wrap')  #  EXCEPTION
        assert x() == 0.0
