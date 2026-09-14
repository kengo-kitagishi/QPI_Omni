try:
    import cupy as xp

    _cp = True

except ImportError:
    import numpy as xp

    _cp = False


def cp2np(array):
    if _cp:
        return xp.asnumpy(array)
    else:
        return array
