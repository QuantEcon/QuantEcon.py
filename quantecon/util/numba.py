"""
Utilities to support Numba jitted functions

"""
import numpy as np
from numba import generated_jit, types
from numba.targets.linalg import _LAPACK


# BLAS kinds as letters
_blas_kinds = {
    types.float32: 's',
    types.float64: 'd',
    types.complex64: 'c',
    types.complex128: 'z',
}


@generated_jit(nopython=True, cache=True)
def _numba_linalg_solve(a, b):
    """
    Solve the linear equation ax = b directly calling a Numba internal
    function. The data in `a` and `b` are interpreted in Fortran order,
    and dtype of `a` and `b` must be the same, one of {float32, float64,
    complex64, complex128}. `a` and `b` are modified in place, and the
    solution is stored in `b`. *No error check is made for the inputs.*

    Parameters
    ----------
    a : ndarray(ndim=2)
        2-dimensional ndarray of shape (n, n).

    b : ndarray(ndim=1 or 2)
        1-dimensional ndarray of shape (n,) or 2-dimensional ndarray of
        shape (n, nrhs).

    Returns
    -------
    r : scalar(int)
        r = 0 if successful.

    Notes
    -----
    From github.com/numba/numba/blob/master/numba/targets/linalg.py

    """
    numba_xgesv = _LAPACK().numba_xgesv(a.dtype)
    kind = ord(_blas_kinds[a.dtype])

    def _numba_linalg_solve_impl(a, b):  # pragma: no cover
        n = a.shape[-1]
        if b.ndim == 1:
            nrhs = 1
        else:  # b.ndim == 2
            nrhs = b.shape[-1]
        F_INT_nptype = np.int32
        ipiv = np.empty(n, dtype=F_INT_nptype)

        r = numba_xgesv(
            kind,         # kind
            n,            # n
            nrhs,         # nhrs
            a.ctypes,     # a
            n,            # lda
            ipiv.ctypes,  # ipiv
            b.ctypes,     # b
            n             # ldb
        )
        return r

    return _numba_linalg_solve_impl
