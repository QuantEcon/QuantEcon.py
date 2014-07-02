"""
Filename: ce_util.py
Authors: Chase Coleman, Spencer Lyon, John Stachurski, and Thomas Sargent
Date: 2014-07-01

Utility functions used in CompEcon

Based routines found in the CompEcon toolbox by Miranda and Fackler.

TODO: Add reference to CompEcon

"""
from functools import reduce
import numpy as np


def ckron(*arrays):
    """
    Repeatedly applies the np.kron function to an arbitrary number of
    input arrays

    Parameters
    ----------
    *arrays : tuple/list of np.ndarray

    Returns
    -------
    out : np.ndarray
        The result of repeated kronecker products

    References
    ----------
    emulates the function ckron.m in the CompEcon toolbox

    """
    return reduce(np.kron, arrays)



def gridmake(*arrays):
    if all([i.ndim == 1 for i in arrays]):
        d = len(arrays)
        if d == 2:
            out = _gridmake2(*arrays)
        else:
            out = _gridmake2(arrays[0], arrays[1])
            for arr in arrays[2:]:
                out = _gridmake2(out, arr)

        return out
    else:
        raise NotImplementedError("Come back here")



def _gridmake2(x1, x2):
    if x1.ndim == 1 and x2.ndim == 1:
        return np.column_stack([np.tile(x1, x2.shape[0]),
                               np.repeat(x2, x1.shape[0])])
    elif x1.ndim > 1 and x2.ndim == 1:
        first = np.tile(x1, (x2.shape[0], 1))
        second = np.repeat(x2, x1.shape[0])
        return np.column_stack([first, second])
    else:
        raise NotImplementedError("Come back here")
