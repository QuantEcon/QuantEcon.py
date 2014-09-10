"""
Filename: test_stochmatrix.py
Author: Daisuke Oyama

Tests for stochmatrix.py

"""
from __future__ import division

import sys
import numpy as np
import nose
from nose.tools import eq_, ok_

from quantecon.stochmatrix import StochMatrix, stationary_dists, gth_solve


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
        self.reducible_matrix_dicts = []
        self.irreducible_matrix_dicts = []

        matrix_dict = {
            'P': np.array([[1, 0], [0, 1]]),
            'comm_classes': [[0], [1]],
            'rec_classes': [[0], [1]],
            'is_irreducible': False,
            }
        self.reducible_matrix_dicts.append(matrix_dict)

        matrix_dict = {
            'P': np.array([[1, 0, 0], [0.5, 0, 0.5], [0, 0, 1]]),
            'comm_classes': [[0], [1], [2]],
            'rec_classes': [[0], [2]],
            'is_irreducible': False,
            }
        self.reducible_matrix_dicts.append(matrix_dict)

        matrix_dict = {
            'P': np.array([[0.4, 0.6], [0.2, 0.8]]),
            'comm_classes': [range(2)],
            'rec_classes': [range(2)],
            'is_irreducible': True,
            }
        self.irreducible_matrix_dicts.append(matrix_dict)

        matrix_dict = {
            'P': KMR_Markov_matrix_sequential(N=27, p=1./3, epsilon=1e-2),
            'comm_classes': [range(27+1)],
            'rec_classes': [range(27+1)],
            'is_irreducible': True,
            }
        self.irreducible_matrix_dicts.append(matrix_dict)

        matrix_dict = {
            'P': KMR_Markov_matrix_sequential(N=3, p=1./3, epsilon=1e-14),
            'comm_classes': [range(3+1)],
            'rec_classes': [range(3+1)],
            'is_irreducible': True,
            }
        self.irreducible_matrix_dicts.append(matrix_dict)

        self.matrix_dicts = \
            self.reducible_matrix_dicts + self.irreducible_matrix_dicts


class TestStochmatrix:
    """Test the methods in StochMatrix"""

    def setUp(self):
        """Setup StochMatrix instances"""
        self.matrices = Matrices()
        for matrix_dict in self.matrices.matrix_dicts:
            matrix_dict['P'] = StochMatrix(matrix_dict['P'])

    def test_comm_classes(self):
        for matrix_dict in self.matrices.matrix_dicts:
            eq_(sorted(matrix_dict['P'].comm_classes()),
                sorted(matrix_dict['comm_classes']))

    def test_num_comm_classes(self):
        for matrix_dict in self.matrices.matrix_dicts:
            eq_(matrix_dict['P'].num_comm_classes,
                len(matrix_dict['comm_classes']))

    def test_rec_classes(self):
        for matrix_dict in self.matrices.matrix_dicts:
            eq_(sorted(matrix_dict['P'].rec_classes()),
                sorted(matrix_dict['rec_classes']))

    def test_num_rec_classes(self):
        for matrix_dict in self.matrices.matrix_dicts:
            eq_(matrix_dict['P'].num_rec_classes,
                len(matrix_dict['rec_classes']))

    def test_is_irreducible(self):
        for matrix_dict in self.matrices.matrix_dicts:
            eq_(matrix_dict['P'].is_irreducible,
                matrix_dict['is_irreducible'])


def test_stationary_dists():
    """Test stationary_dists"""
    print(__name__ + '.' + test_stationary_dists.__name__)
    matrices = Matrices()
    for matrix_dict in matrices.matrix_dicts:
        dists = stationary_dists(matrix_dict['P'])
        yield NumStationaryDists(), dists, len(matrix_dict['rec_classes'])
        for x in dists:
            yield StationaryDistSumOne(), x
            yield StationaryDistNonnegative(), x
            yield StationaryDistLeftEigenVec(), matrix_dict['P'], x


def test_gth_solve():
    """Test gth_solve"""
    print(__name__ + '.' + test_gth_solve.__name__)
    matrices = Matrices()
    # Use only irreducible matrices
    for matrix_dict in matrices.irreducible_matrix_dicts:
        x = gth_solve(matrix_dict['P'])
        yield StationaryDistSumOne(), x
        yield StationaryDistNonnegative(), x
        yield StationaryDistLeftEigenVec(), matrix_dict['P'], x


class AddDescription:
    def __init__(self):
        self.description = self.__class__.__name__


class NumStationaryDists(AddDescription):
    def __call__(self, dists, n):
        eq_(len(dists), n)


class StationaryDistSumOne(AddDescription):
    def __call__(self, x):
        ok_(np.allclose(sum(x), 1, atol=TOL))


class StationaryDistNonnegative(AddDescription):
    def __call__(self, x):
        eq_(np.prod(x >= 0-TOL), 1)


class StationaryDistLeftEigenVec(AddDescription):
    def __call__(self, P, x):
        ok_(np.allclose(np.dot(x, P), x, atol=TOL))


if __name__ == '__main__':
    argv = sys.argv[:]
    argv.append('--verbose')
    argv.append('--nocapture')
    nose.main(argv=argv, defaultTest=__file__)
