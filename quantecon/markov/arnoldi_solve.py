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
        If an explicit matrix is not finite, real, nonnegative, and
        row-stochastic, or if the eigenvalue is nonfinite or outside
        `sqrt(max(tol, eps))` of 1. A `LinearOperator` is assumed to
        represent a stochastic matrix; the eigenvalue check is not a
        validation of its entries or row sums.

    scipy.sparse.linalg.ArpackNoConvergence
        If the iteration does not converge within `max_iter` restarts.

    Notes
    -----
    `gth_solve` computes the stationary distribution of an irreducible
    stochastic matrix by the GTH algorithm, a variant of Gaussian
    elimination without subtractions; it requires a dense matrix.
    `arnoldi_solve` wraps a general-purpose iterative solver that acts on
    the matrix, or the operator, as given, so it is suited to chains too
    large to densify. Explicit row sums are accumulated in at least
    double precision and checked within the larger of `sqrt(eps64)`
    and eight epsilons of ARPACK's working dtype, independently of
    the eigensolver tolerance. Validation never densifies sparse
    input. Noncanonical or uncommon sparse formats may need sparse
    scratch space; the solver still uses the original matrix/operator.

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
        P = np.asarray(P)  # P.T is a view: P is never copied
    if len(P.shape) != 2 or P.shape[0] != P.shape[1]:
        raise ValueError('matrix must be square')
    op, n = P.T, P.shape[0]
    if n == 0:
        raise ValueError('matrix must be nonempty')
    dtype = _working_dtype(P.dtype)
    if not isinstance(P, LinearOperator):
        _check_explicit_stochastic(P, dtype)

    if x0 is None:
        x0 = np.full(n, 1 / n, dtype=dtype)
    else:
        x0 = np.array(x0, dtype=dtype)  # copy: normalized in place below
        if x0.shape != (n,):
            raise ValueError(f'x0 must have shape ({n},); got {x0.shape}')
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

    if P.dtype.char not in 'fd':
        # ARPACK takes float32 and float64 only: the products of a matrix
        # of another dtype are converted here
        Pt = op
        op = LinearOperator(
            (n, n), dtype=dtype,
            matvec=lambda x: np.asarray(Pt @ x, dtype=dtype)
        )

    vals, vecs = eigs(op, k=1, which='LR', v0=x0, tol=tol, maxiter=max_iter,
                      ncv=ncv)
    _check_eigenvalue(vals[0], slack)

    # the eigenvector is complex and has arbitrary phase and scale, and
    # entries that are zero in exact arithmetic (at transient states) may
    # come out negative at rounding level. Moreover, for a reducible
    # chain the eigenvalue 1 is multiple, and when the Krylov space is
    # nearly invariant, the normalized rounding noise mixes in
    # eigenvectors with opposite signs on different recurrent classes.
    # The absolute value fixes the phase and the signs: for an
    # eigenvector v of a stochastic operator, |v| is an eigenvector too
    # (A|v| >= |Av| = |v| entrywise, with equality as A preserves the
    # sum), and normalizing by the sum fixes the scale
    x = np.abs(vecs[:, 0])
    return x / x.sum()


def _check_explicit_stochastic(P, dtype):
    if not np.issubdtype(P.dtype, np.number) and P.dtype.kind != 'b':
        raise ValueError('the transition matrix must be real numeric')
    if P.dtype.kind == 'c':
        raise ValueError('the transition matrix must be real numeric')
    if issparse(P):
        entries = P if P.format in ('csr', 'csc', 'coo') else P.tocoo()
        if not np.isfinite(entries.data).all():
            raise ValueError('the transition matrix must be finite')
        if (entries.data < 0).any():
            # A signed stored term is not necessarily a negative matrix
            # entry when repeated coordinates have not been coalesced.
            if not entries.has_canonical_format:
                entries = entries.tocoo(copy=True)
                entries = entries.astype(np.result_type(entries.dtype, np.float64),
                                         copy=False)
                entries.sum_duplicates()
            if (entries.data < 0).any():
                raise ValueError('the transition matrix must be nonnegative')
        rows = P @ np.ones(P.shape[1], dtype=np.float64)
    else:
        if not np.isfinite(P).all():
            raise ValueError('the transition matrix must be finite')
        if (P < 0).any():
            raise ValueError('the transition matrix must be nonnegative')
        rows = P.sum(axis=1, dtype=np.result_type(P.dtype, np.float64))
    slack = max(np.sqrt(np.finfo(np.float64).eps), 8 * np.finfo(dtype).eps)
    if not np.all(np.abs(rows - 1) <= slack):
        raise ValueError('the matrix is not stochastic (rows must sum to 1)')


def _check_eigenvalue(lam, slack):
    if not np.isfinite(lam) or abs(lam - 1) > slack:
        raise ValueError(
            f'the largest eigenvalue {lam} is not 1: the matrix is not '
            'stochastic (rows must sum to 1)'
        )


def _working_dtype(dtype):
    """
    Working precision: ARPACK supports float32 and float64 only.

    """
    return np.float32 if dtype.char == 'f' else np.float64


def _solve_2x2(op, x0, slack, dtype):
    """
    Stationary distribution of a 2-state chain in closed form; `op` is
    the transposed transition matrix. With two absorbing states, `x0`
    (which is then stationary) is returned.

    """
    Pt = np.asarray(op @ np.eye(2, dtype=dtype), dtype=dtype)
    if not np.isfinite(Pt).all() or (Pt < 0).any():
        raise ValueError('the transition matrix must be finite and nonnegative')
    if np.abs(Pt.sum(axis=0) - 1).max() > slack:
        raise ValueError('the matrix is not stochastic (rows must sum to 1)')
    a, b = Pt[1, 0], Pt[0, 1]  # P[0, 1] and P[1, 0]
    if a + b == 0:
        return x0
    return np.array([b, a], dtype=dtype) / (a + b)
