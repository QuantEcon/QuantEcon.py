"""
Tests for gth_solve.py

"""
import numpy as np
from numpy.testing import (assert_array_equal, assert_raises, assert_,
                           assert_allclose)

from quantecon.markov import gth_solve
import pytest


TOL = 1e-15


def KMR_Markov_matrix_sequential(N, p, epsilon):
    """
    Generate the Markov matrix for the KMR model with *sequential* move

    Parameters
    ----------
    N : int
        Number of players

    p : float
        Level of p-dominance of action 1, i.e.,
        the value of p such that action 1 is the BR for (1-q, q) for any q > p,
        where q (1-q, resp.) is the prob that the opponent plays action 1 (0, resp.)

    epsilon : float
        Probability of mutation

    Returns
    -------
    P : numpy.ndarray
        Markov matrix for the KMR model with simultaneous move

    """
    P = np.zeros((N+1, N+1), dtype=float)
    P[0, 0], P[0, 1] = 1 - epsilon * (1/2), epsilon * (1/2)
    for n in range(1, N):
        P[n, n-1] = \
            (n/N) * (epsilon * (1/2) +
                     (1 - epsilon) * (((n-1)/(N-1) < p) + ((n-1)/(N-1) == p) * (1/2))
                     )
        P[n, n+1] = \
            ((N-n)/N) * (epsilon * (1/2) +
                         (1 - epsilon) * ((n/(N-1) > p) + (n/(N-1) == p) * (1/2))
                         )
        P[n, n] = 1 - P[n, n-1] - P[n, n+1]
    P[N, N-1], P[N, N] = epsilon * (1/2), 1 - epsilon * (1/2)
    return P


class Matrices:
    """Setup matrices for the tests"""

    def __init__(self):
        self.stoch_matrix_dicts = []
        self.kmr_matrix_dicts = []
        self.gen_matrix_dicts = []

        matrix_dict = {
            'A': np.array([[0.4, 0.6], [0.2, 0.8]]),
            'stationary_dist': np.array([0.25, 0.75]),
        }
        self.stoch_matrix_dicts.append(matrix_dict)

        matrix_dict = {
            # Reducible matrix
            'A': np.array([[1, 0], [0, 1]]),
            # Stationary dist whose support contains index 0
            'stationary_dist': np.array([1, 0]),
        }
        self.stoch_matrix_dicts.append(matrix_dict)

        matrix_dict = {
            'A': KMR_Markov_matrix_sequential(N=27, p=1./3, epsilon=1e-2),
        }
        self.kmr_matrix_dicts.append(matrix_dict)

        matrix_dict = {
            'A': KMR_Markov_matrix_sequential(N=3, p=1./3, epsilon=1e-14),
        }
        self.kmr_matrix_dicts.append(matrix_dict)

        matrix_dict = {
            'A': np.array([[-3, 3], [4, -4]]),
            'stationary_dist': np.array([4/7, 3/7]),
        }
        self.gen_matrix_dicts.append(matrix_dict)


class AddDescription:
    def __init__(self):
        self.description = self.__class__.__name__


class StationaryDistSumOne(AddDescription):
    def __call__(self, x):
        assert_allclose(np.sum(x), 1, atol=TOL)


class StationaryDistNonnegative(AddDescription):
    def __call__(self, x):
        assert_(np.prod(x >= 0-TOL) == 1)


class StationaryDistLeftEigenVec(AddDescription):
    def __call__(self, A, x):
        assert_allclose(np.dot(x, A), x, atol=TOL)


class StationaryDistEqualToKnown(AddDescription):
    def __call__(self, y, x):
        assert_allclose(y, x, atol=TOL)


test_classes = [
    StationaryDistSumOne,
    StationaryDistNonnegative,
]


@pytest.mark.parametrize("test_class", test_classes)
def test_stoch_matrix(test_class):
    """Test with stochastic matrices"""
    matrices = Matrices()
    for matrix_dict in matrices.stoch_matrix_dicts:
        x = gth_solve(matrix_dict['A'])
        test_class()(x)


def test_stoch_matrix_1():
    """Test with stochastic matrices"""
    matrices = Matrices()
    for matrix_dict in matrices.stoch_matrix_dicts:
        x = gth_solve(matrix_dict['A'])
        StationaryDistEqualToKnown()(matrix_dict['stationary_dist'], x)


@pytest.mark.parametrize("test_class", test_classes)
def test_kmr_matrix(test_class):
    """Test with KMR matrices"""
    matrices = Matrices()
    for matrix_dict in matrices.kmr_matrix_dicts:
        x = gth_solve(matrix_dict['A'])
        test_class()(x)


def test_kmr_matrix_1():
    """Test with KMR matrices"""
    matrices = Matrices()
    for matrix_dict in matrices.kmr_matrix_dicts:
        x = gth_solve(matrix_dict['A'])
        StationaryDistLeftEigenVec()(matrix_dict['A'], x)


@pytest.mark.parametrize("test_class", test_classes)
def test_gen_matrix(test_class):
    """Test with generator matrices"""
    matrices = Matrices()
    for matrix_dict in matrices.gen_matrix_dicts:
        x = gth_solve(matrix_dict['A'])
        test_class()(x)


def test_gen_matrix_1():
    """Test with generator matrices"""
    matrices = Matrices()
    for matrix_dict in matrices.gen_matrix_dicts:
        x = gth_solve(matrix_dict['A'])
        StationaryDistEqualToKnown()(matrix_dict['stationary_dist'], x)


def test_matrices_with_C_F_orders():
    """
    Test matrices with C- and F-contiguous orders
    See the issue and fix on Numba:
    github.com/numba/numba/issues/1103
    github.com/numba/numba/issues/1104

    Fix in gth_solve(A):
    added `order='C'` when `A1` copies the input `A`
    """
    P_C = np.array([[0.5, 0.5], [0, 1]], order='C')
    P_F = np.array([[0.5, 0.5], [0, 1]], order='F')
    stationary_dist = [0., 1.]

    computed_C_and_F = gth_solve(np.array([[1]]))
    assert_array_equal(computed_C_and_F, [1])

    computed_C = gth_solve(P_C)
    computed_F = gth_solve(P_F)
    assert_array_equal(computed_C, stationary_dist)
    assert_array_equal(computed_F, stationary_dist)


def test_raises_value_error_non_2dim():
    """Test with non 2dim input"""
    assert_raises(ValueError, gth_solve, np.array([0.4, 0.6]))


def test_raises_value_error_non_square():
    """Test with non square input"""
    assert_raises(ValueError, gth_solve, np.array([[0.4, 0.6]]))
