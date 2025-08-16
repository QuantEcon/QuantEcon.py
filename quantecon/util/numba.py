"""
Utilities to support Numba jitted functions

"""
import numpy as np
from numba import jit, types
from numba.extending import overload
from numba.np.linalg import _LAPACK


# BLAS kinds as letters
_blas_kinds = {
    types.float32: 's',
    types.float64: 'd',
    types.complex64: 'c',
    types.complex128: 'z',
}


def _numba_linalg_solve(a, b):  # pragma: no cover
    pass


@overload(_numba_linalg_solve, jit_options={'cache':True})
def _numba_linalg_solve_ol(a, b):
    """
    Solve the linear equation ax = b directly calling a Numba internal
    function. The data in `a` and `b` are interpreted in Fortran order,
    and dtype of `a` and `b` must be the same, one of {float32, float64,
    complex64, complex128}. `a` and `b` are modified in place, and the
    solution is stored in `b`. *No error check is made for the inputs.*
    Only work in a Numba-jitted function.

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
    From github.com/numba/numba/blob/main/numba/np/linalg.py

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


@jit(types.intp(types.intp, types.intp), nopython=True, cache=True)
def comb_jit(N, k):
    """
    Numba jitted function that computes N choose k. Return `0` if the
    outcome exceeds the maximum value of `np.intp` or if N < 0, k < 0,
    or k > N.

    Parameters
    ----------
    N : scalar(int)

    k : scalar(int)

    Returns
    -------
    val : scalar(int)

    """
    # From scipy.special._comb_int_long
    # github.com/scipy/scipy/blob/v1.0.0/scipy/special/_comb.pyx
    INTP_MAX = np.iinfo(np.intp).max
    if N < 0 or k < 0 or k > N:
        return 0
    if k == 0:
        return 1
    if k == 1:
        return N
    if N == INTP_MAX:
        return 0

    M = N + 1
    nterms = min(k, N - k)

    val = 1

    for j in range(1, nterms + 1):
        # Overflow check
        if val > INTP_MAX // (M - j):
            return 0

        val *= M - j
        val //= j

    return val
