import numpy as np
from numpy.linalg import svd


def rank_est(A, atol=1e-13, rtol=0):
    """
    Estimate the rank (i.e. the dimension of the nullspace) of a matrix.

    The algorithm used by this function is based on the singular value
    decomposition of `A`.

    Parameters
    ----------
    A : array_like(float, ndim=1 or 2)
        A should be at most 2-D.  A 1-D array with length n will be
        treated as a 2-D with shape (1, n)
    atol : scalar(float), optional(default=1e-13)
        The absolute tolerance for a zero singular value.  Singular
        values smaller than `atol` are considered to be zero.
    rtol : scalar(float), optional(default=0)
        The relative tolerance.  Singular values less than rtol*smax are
        considered to be zero, where smax is the largest singular value.

    Returns
    -------
    r : scalar(int)
        The estimated rank of the matrix.

    Note: If both `atol` and `rtol` are positive, the combined tolerance
    is the maximum of the two; that is:

        tol = max(atol, rtol * smax)

    Note: Singular values smaller than `tol` are considered to be zero.

    See also
    --------
    numpy.linalg.matrix_rank
        matrix_rank is basically the same as this function, but it does
        not provide the option of the absolute tolerance.

    """

    A = np.atleast_2d(A)
    s = svd(A, compute_uv=False)
    tol = max(atol, rtol * s[0])
    rank = int((s >= tol).sum())

    return rank


def nullspace(A, atol=1e-13, rtol=0):
    """
    Compute an approximate basis for the nullspace of A.

    The algorithm used by this function is based on the singular value
    decomposition of `A`.

    Parameters
    ----------
    A : array_like(float, ndim=1 or 2)
        A should be at most 2-D.  A 1-D array with length k will be
        treated as a 2-D with shape (1, k)
    atol : scalar(float), optional(default=1e-13)
        The absolute tolerance for a zero singular value.  Singular
        values smaller than `atol` are considered to be zero.
    rtol : scalar(float), optional(default=0)
        The relative tolerance.  Singular values less than rtol*smax are
        considered to be zero, where smax is the largest singular value.

    Returns
    -------
    ns : array_like(float, ndim=2)
        If `A` is an array with shape (m, k), then `ns` will be an array
        with shape (k, n), where n is the estimated dimension of the
        nullspace of `A`.  The columns of `ns` are a basis for the
        nullspace; each element in numpy.dot(A, ns) will be
        approximately zero.

    Note: If both `atol` and `rtol` are positive, the combined tolerance
    is the maximum of the two; that is:

        tol = max(atol, rtol * smax)

    Note: Singular values smaller than `tol` are considered to be zero.

    """

    A = np.atleast_2d(A)
    u, s, vh = svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T

    return ns
