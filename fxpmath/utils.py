
def twos_complement(val, nbits):
    if val < 0:
        val = (1 << nbits) + val
    else:
        val = val % (1 << nbits) 
        if (val & (1 << (nbits - 1))) != 0:
            val = val - (1 << nbits)
    return val