"""
Routine to compute the stationary distribution of an irreducible Markov
chain by the Grassmann-Taksar-Heyman (GTH) algorithm.

"""
import numpy as np
from numba import jit

def gth_solve(A, overwrite=False, use_jit=True):
    r"""
    This routine computes the stationary distribution of an irreducible
    Markov transition matrix (stochastic matrix) or transition rate
    matrix (generator matrix) `A`.

    More generally, given a Metzler matrix (square matrix whose
    off-diagonal entries are all nonnegative) `A`, this routine solves
    for a nonzero solution `x` to `x (A - D) = 0`, where `D` is the
    diagonal matrix for which the rows of `A - D` sum to zero (i.e.,
    :math:`D_{ii} = \sum_j A_{ij}` for all :math:`i`). One (and only
    one, up to normalization) nonzero solution exists corresponding to
    each reccurent class of `A`, and in particular, if `A` is
    irreducible, there is a unique solution; when there are more than
    one solution, the routine returns the solution that contains in its
    support the first index `i` such that no path connects `i` to any
    index larger than `i`. The solution is normalized so that its 1-norm
    equals one. This routine implements the Grassmann-Taksar-Heyman
    (GTH) algorithm [1]_, a numerically stable variant of Gaussian
    elimination, where only the off-diagonal entries of `A` are used as
    the input data. For a nice exposition of the algorithm, see Stewart
    [2]_, Chapter 10.

    Parameters
    ----------
    A : array_like(float, ndim=2)
        Stochastic matrix or generator matrix. Must be of shape n x n.

    Returns
    -------
    x : numpy.ndarray(float, ndim=1)
        Stationary distribution of `A`.

    overwrite : bool, optional(default=False)
        Whether to overwrite `A`.

    References
    ----------
    .. [1] W. K. Grassmann, M. I. Taksar and D. P. Heyman, "Regenerative
       Analysis and Steady State Distributions for Markov Chains,"
       Operations Research (1985), 1107-1116.

    .. [2] W. J. Stewart, Probability, Markov Chains, Queues, and
       Simulation, Princeton University Press, 2009.

    """
    A1 = np.array(A, dtype=float, copy=not overwrite, order='C')
    # `order='C'` is for use with Numba <= 0.18.2
    # See issue github.com/numba/numba/issues/1103

    if len(A1.shape) != 2 or A1.shape[0] != A1.shape[1]:
        raise ValueError('matrix must be square')

    n = A1.shape[0]
    x = np.zeros(n)

    if use_jit:
        _gth_solve_jit(A1, x)
        return x

    # if not using jit
    # === Reduction === #
    for k in range(n-1):
        scale = np.sum(A1[k, k+1:n])
        if scale <= 0:
            # There is one (and only one) recurrent class contained in
            # {0, ..., k};
            # compute the solution associated with that recurrent class.
            n = k+1
            break
        A1[k+1:n, k] /= scale

        A1[k+1:n, k+1:n] += np.dot(A1[k+1:n, k:k+1], A1[k:k+1, k+1:n])

    # === Backward substitution === #
    x[n-1] = 1
    for k in range(n-2, -1, -1):
        x[k] = np.dot(x[k+1:n], A1[k+1:n, k])

    # === Normalization === #
    x /= np.sum(x)

    return x


@jit(nopython=True)
def _gth_solve_jit(A, out):
    """
    JIT complied version of the main routine of gth_solve.

    Parameters
    ----------
    A : numpy.ndarray(float, ndim=2)
        Stochastic matrix or generator matrix. Must be of shape n x n.
        Data will be overwritten.

    out : numpy.ndarray(float, ndim=1)
        Output array in which to place the stationary distribution of A.

    """
    n = A.shape[0]

    # === Reduction === #
    for k in range(n-1):
        scale = np.sum(A[k, k+1:n])
        if scale <= 0:
            # There is one (and only one) recurrent class contained in
            # {0, ..., k};
            # compute the solution associated with that recurrent class.
            n = k+1
            break
        for i in range(k+1, n):
            A[i, k] /= scale

            for j in range(k+1, n):
                A[i, j] += A[i, k] * A[k, j]

    # === Backward substitution === #
    out[n-1] = 1
    for k in range(n-2, -1, -1):
        for i in range(k+1, n):
            out[k] += out[i] * A[i, k]

    # === Normalization === #
    norm = np.sum(out)
    for k in range(n):
        out[k] /= norm
