"""
Tests for arnoldi_solve.py

"""
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal, assert_raises
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, ArpackNoConvergence
import pytest

from quantecon.markov import arnoldi_solve, gth_solve


def random_stochastic_matrix(rng, n):
    P = rng.random((n, n))
    P /= P.sum(axis=1, keepdims=True)
    return P


def random_sparse_stochastic_matrix(rng, n, k):
    """
    Sparse stochastic matrix with k nonzeros per row, irreducible since
    each row i contains the entry (i, (i+1) % n).

    """
    rows = np.repeat(np.arange(n), k)
    cols = np.empty((n, k), dtype=int)
    for i in range(n):
        others = rng.choice(np.delete(np.arange(n), (i+1) % n), k-1,
                            replace=False)
        cols[i] = np.r_[(i+1) % n, others]
    vals = rng.random((n, k))
    vals /= vals.sum(axis=1, keepdims=True)
    return sparse.csr_array((vals.ravel(), (rows, cols.ravel())),
                            shape=(n, n))


def as_linear_operator(P):
    return LinearOperator(P.shape, matvec=lambda v: P @ v,
                          rmatvec=lambda x: P.T @ x, dtype=P.dtype)


# Ways of passing the same chain
input_kinds = {
    'dense': lambda P: P,
    'csr_matrix': sparse.csr_matrix,
    'csr_array': sparse.csr_array,
    'linear_operator': as_linear_operator,
}


@pytest.mark.parametrize('kind', input_kinds)
def test_irreducible_dense_chain(kind):
    rng = np.random.default_rng(0)
    P = random_stochastic_matrix(rng, 20)
    ref = gth_solve(P)
    x = arnoldi_solve(input_kinds[kind](P), np.full(20, 1/20))
    assert x.dtype == np.float64
    assert x.shape == (20,)
    assert_allclose(x.sum(), 1, atol=1e-15)
    assert_allclose(x, ref, atol=1e-10)
    # any starting vector gives the unique stationary distribution
    assert_allclose(arnoldi_solve(input_kinds[kind](P), np.eye(20)[0]),
                    ref, atol=1e-10)


def test_default_x0():
    rng = np.random.default_rng(1)
    P = random_stochastic_matrix(rng, 20)
    assert_allclose(arnoldi_solve(P), gth_solve(P), atol=1e-10)
    assert_allclose(arnoldi_solve(sparse.csr_array(P)), gth_solve(P),
                    atol=1e-10)


def test_ncv():
    rng = np.random.default_rng(3)
    P = random_stochastic_matrix(rng, 20)
    assert_allclose(arnoldi_solve(P, ncv=5), gth_solve(P), atol=1e-10)
    assert_raises(ValueError, arnoldi_solve, P, ncv=2)  # too small


def test_irreducible_sparse_chain():
    rng = np.random.default_rng(2)
    P = random_sparse_stochastic_matrix(rng, 200, 5)
    ref = gth_solve(P.toarray())
    x = arnoldi_solve(P)
    assert_allclose(x, ref, atol=1e-10)
    assert_allclose(arnoldi_solve(sparse.csr_matrix(P)), ref, atol=1e-10)
    assert_allclose(arnoldi_solve(P.T.T), ref, atol=1e-10)  # csc -> csr


def test_linear_operator_without_rmatvec():
    P = np.array([[0.9, 0.1, 0.0], [0.0, 0.8, 0.2], [0.5, 0.0, 0.5]])
    op = LinearOperator(P.shape, matvec=lambda v: P @ v, dtype=float)
    assert_raises(NotImplementedError, arnoldi_solve, op)


def test_symmetric_chain():
    # random walk on a 3-regular graph: stationary = uniform
    n = 12
    P = np.zeros((n, n))
    for i in range(n):
        P[i, (i+1) % n] += 1/3
        P[i, (i-1) % n] += 1/3
        P[i, (i+n//2) % n] += 1/3
    assert_allclose(arnoldi_solve(P), np.full(n, 1/n), atol=1e-10)


def test_periodic_chain():
    # cycle of length 5: eigenvalues are the fifth roots of unity
    P = np.roll(np.eye(5), 1, axis=1)
    assert_allclose(arnoldi_solve(P), np.full(5, 0.2), atol=1e-10)
    assert_allclose(arnoldi_solve(P, np.eye(5)[0]), np.full(5, 0.2),
                    atol=1e-10)


def test_nearly_decomposable_chain():
    # two blocks coupled with probability 1e-6: the spectral gap is
    # tiny, where power iteration would stall; the error is of the order
    # of the residual divided by the gap
    rng = np.random.default_rng(4)
    n = 20
    P = np.zeros((2*n, 2*n))
    P[:n, :n] = random_stochastic_matrix(rng, n)
    P[n:, n:] = random_stochastic_matrix(rng, n)
    eps_c = 1e-6
    P[0, 0] -= eps_c
    P[0, n] += eps_c
    P[n, n] -= eps_c
    P[n, 0] += eps_c
    assert_allclose(arnoldi_solve(P), gth_solve(P), atol=1e-8)


def test_float32():
    rng = np.random.default_rng(5)
    P = random_stochastic_matrix(rng, 20).astype(np.float32)
    P /= P.sum(axis=1, keepdims=True)
    x = arnoldi_solve(P)
    assert x.dtype == np.float32
    assert_allclose(x, gth_solve(P.astype(np.float64)), atol=1e-4)
    # float32 P with float64 x0: P decides
    assert arnoldi_solve(P, np.full(20, 1/20)).dtype == np.float32


def test_other_dtypes():
    # any real matrix is used as given; the computation is in float64
    Q = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])  # deterministic 3-cycle
    x = arnoldi_solve(Q)
    assert x.dtype == np.float64
    assert_allclose(x, np.full(3, 1/3), atol=1e-10)
    P = np.array([[0.5, 0.5, 0], [0.25, 0.5, 0.25], [0, 0.5, 0.5]])
    ref = gth_solve(P)
    for dtype in (np.float16, np.longdouble):
        y = arnoldi_solve(P.astype(dtype))
        assert y.dtype == np.float64
        assert_allclose(y, ref, atol=1e-10)
    assert_allclose(arnoldi_solve(P, np.array([1, 0, 0], dtype=np.longdouble)),
                    ref, atol=1e-10)


def test_small_chains():
    assert_array_equal(arnoldi_solve(np.ones((1, 1))), [1.])
    assert_array_equal(arnoldi_solve([[1.]], [3.]), [1.])
    P = np.array([[0.3, 0.7], [0.6, 0.4]])
    ref = gth_solve(P)
    for kind in input_kinds:
        assert_allclose(arnoldi_solve(input_kinds[kind](P), [0.5, 0.5]),
                        ref, atol=1e-15)
    assert_allclose(arnoldi_solve(P), ref, atol=1e-15)
    # one absorbing state
    assert_allclose(arnoldi_solve([[0.5, 0.5], [0, 1]]), [0, 1], atol=1e-15)
    # float32 result for a float32 matrix
    assert arnoldi_solve(P.astype(np.float32)).dtype == np.float32
    # not stochastic
    assert_raises(ValueError, arnoldi_solve, 0.9 * P)


def test_errors():
    rng = np.random.default_rng(6)
    P = random_stochastic_matrix(rng, 10)
    # not stochastic: the largest eigenvalue is 0.9
    assert_raises(ValueError, arnoldi_solve, 0.9 * P)
    # starting vector
    assert_raises(ValueError, arnoldi_solve, P, np.ones(9))
    assert_raises(ValueError, arnoldi_solve, P, np.zeros(10))
    assert_raises(ValueError, arnoldi_solve, P, np.r_[-1., np.ones(9)])
    assert_raises(ValueError, arnoldi_solve, P, np.r_[np.nan, np.ones(9)])
    assert_raises(ValueError, arnoldi_solve, P, np.r_[np.inf, np.ones(9)])
    # non-square, non-2d
    assert_raises(ValueError, arnoldi_solve, np.ones((2, 3)) / 3)
    assert_raises(ValueError, arnoldi_solve, np.array([0.4, 0.6]))
    assert_raises(ValueError, arnoldi_solve, sparse.csr_array(np.ones((2, 3))))
    # non-convergence: a nearly decomposable chain with one restart of
    # a three-dimensional Krylov space
    n = 20
    Pnd = np.zeros((2*n, 2*n))
    Pnd[:n, :n] = random_stochastic_matrix(rng, n)
    Pnd[n:, n:] = random_stochastic_matrix(rng, n)
    Pnd[0, 0] -= 1e-4
    Pnd[0, n] += 1e-4
    Pnd[n, n] -= 1e-4
    Pnd[n, 0] += 1e-4
    assert_raises(ArpackNoConvergence, arnoldi_solve, Pnd, max_iter=1,
                  ncv=3)


def test_x0_not_modified():
    P = np.array([[0.9, 0.1, 0.0], [0.0, 0.8, 0.2], [0.5, 0.0, 0.5]])
    x0 = np.array([3., 0., 0.])
    arnoldi_solve(P, x0)
    assert_array_equal(x0, [3., 0., 0.])
