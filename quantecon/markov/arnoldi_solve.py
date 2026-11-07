"""
Routine to compute a stationary distribution of a Markov chain by
restarted Arnoldi iteration (ARPACK), for chains whose transition matrix
is too large for the GTH algorithm.

"""
import numpy as np
from scipy.sparse import issparse
from scipy.sparse.linalg import eigs, LinearOperator


def arnoldi_solve(P, x0=None, tol=None, max_iter=None, ncv=None):
    r"""
    Compute a stationary distribution of a Markov chain with `n` states
    by restarted Arnoldi iteration on the push-forward operator
    :math:`x \mapsto x P`, using ARPACK through `scipy.sparse.linalg.eigs
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigs.html>`_.

    The chain is given as its transition matrix `P`, whose rows are the
    transition probabilities from each state. Its transpose is passed to
    `scipy.sparse.linalg.eigs` to solve for the eigenvalue of largest
    real part, which is 1 for a stochastic matrix; `tol`, `max_iter`,
    and `ncv` are passed to it as `tol`, `maxiter`, and `ncv`.

    ARPACK works in double or single precision, so the returned array
    has dtype `float64`, unless `P` has dtype `float32`, in which case
    it has dtype `float32`.

    Parameters
    ----------
    P : array_like, sparse array or matrix, or LinearOperator
        Stochastic matrix of shape n x n. A `LinearOperator` must
        implement `rmatvec`, which computes `x @ P`.

    x0 : array_like(float, ndim=1), optional(default=None)
        Starting vector, nonnegative with positive sum (normalized
        internally). If None, the uniform distribution.

    tol : scalar(float), optional(default=None)
        Relative accuracy for the eigenvalue (stopping criterion). If
        None, machine precision.

    max_iter : scalar(int), optional(default=None)
        Maximum number of restarts of the Arnoldi iteration. If None,
        SciPy's default, `10 * n`.

    ncv : scalar(int), optional(default=None)
        Number of Arnoldi vectors generated per restart; larger values
        use more memory but converge in fewer restarts. If None, SciPy's
        default, `min(n, 20)`.

    Returns
    -------
    x : ndarray(float, ndim=1)
        Stationary distribution of the chain.

    Raises
    ------
    ValueError
        If the eigenvalue found is not 1 within `sqrt(max(tol, eps))`,
        which means that the matrix is not stochastic.

    scipy.sparse.linalg.ArpackNoConvergence
        If the iteration does not converge within `max_iter` restarts.

    Notes
    -----
    `gth_solve` computes the stationary distribution of an irreducible
    stochastic matrix by the GTH algorithm, a variant of Gaussian
    elimination without subtractions; it converts the matrix to a dense
    array, even for a sparse input. `arnoldi_solve` wraps a
    general-purpose iterative solver that acts on the matrix, or the
    operator, as given, so it is suited to chains too large to densify.

    Examples
    --------
    >>> P = np.array([[0.9, 0.1, 0.0], [0.0, 0.8, 0.2], [0.5, 0.0, 0.5]])
    >>> x = arnoldi_solve(P)
    >>> x
    array([0.58823529, 0.29411765, 0.11764706])
    >>> np.allclose(x @ P, x)
    True

    """
    if not (isinstance(P, LinearOperator) or issparse(P)):
        # P.T is a view; ARPACK upcasts integer and other dtypes to
        # float64 in its products, without copying P
        P = np.asarray(P)
    if len(P.shape) != 2 or P.shape[0] != P.shape[1]:
        raise ValueError('matrix must be square')
    op, n = P.T, P.shape[0]
    dtype = _working_dtype(P.dtype)

    if x0 is None:
        x0 = np.full(n, 1 / n, dtype=dtype)
    else:
        x0 = np.array(x0, dtype=dtype)  # copy: normalized in place below
        if x0.shape != (n,):
            raise ValueError(f'x0 must have shape ({n},); got {x0.shape}')
    if n == 0:
        raise ValueError('matrix must be nonempty')
    if not np.all(x0 >= 0):
        raise ValueError('x0 must be nonnegative')
    s = x0.sum()
    if not (np.isfinite(s) and s > 0):
        raise ValueError('x0 must have positive finite sum')
    x0 /= s

    if tol is None:
        tol = 0  # SciPy's default: machine precision
    # slack for the check that the eigenvalue found is 1: the error of a
    # converged eigenvalue is of the order of tol (relative to 1)
    slack = np.sqrt(max(tol, np.finfo(dtype).eps))

    if n == 1:
        # the operator is the scalar eigenvalue itself
        lam = np.asarray(op @ x0, dtype=dtype).ravel()[0]
        _check_eigenvalue(lam, slack)
        return np.ones(1, dtype=dtype)
    if n == 2:
        # ARPACK needs k < n - 1; the 2 x 2 case is solved in closed form
        return _solve_2x2(op, x0, slack, dtype)

    vals, vecs = eigs(op, k=1, which='LR', v0=x0, tol=tol, maxiter=max_iter,
                      ncv=ncv)
    _check_eigenvalue(vals[0], slack)

    # the eigenvector is complex (with zero imaginary part at a real
    # eigenvalue) and has arbitrary sign and scale; normalizing by the
    # sum fixes both. Entries that are negative at rounding level are
    # left as they are, as is the convention for eigenvector-based
    # solvers
    x = vecs[:, 0].real
    return x / x.sum()


def _check_eigenvalue(lam, slack):
    if abs(lam - 1) > slack:
        raise ValueError(
            f'the largest eigenvalue {lam} is not 1: the matrix is not '
            'stochastic (rows must sum to 1)'
        )


def _working_dtype(dtype):
    """
    Working precision: ARPACK supports float32 and float64 only.

    """
    return np.float32 if dtype == np.float32 else np.float64


def _solve_2x2(op, x0, slack, dtype):
    """
    Stationary distribution of a 2-state chain in closed form; `op` is
    the transposed transition matrix. With two absorbing states, `x0`
    (which is then stationary) is returned.

    """
    Pt = np.asarray(op @ np.eye(2, dtype=dtype), dtype=dtype)
    if np.abs(Pt.sum(axis=0) - 1).max() > slack:
        raise ValueError('the matrix is not stochastic (rows must sum to 1)')
    a, b = Pt[1, 0], Pt[0, 1]  # P[0, 1] and P[1, 0]
    if a + b == 0:
        return x0
    return np.array([b, a], dtype=dtype) / (a + b)
