"""
Tests for mc_tools.py

Functions
---------
    mc_compute_stationary   [Status: Tested in test_markovchain_pmatrices]
    mc_sample_path          [Status: TBD]

"""
from __future__ import division

import sys
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import nose
from nose.tools import raises

from quantecon.mc_tools import MarkovChain, mc_compute_stationary


# KMR Function
# Useful because it seems to have 1 unit eigvalue, but a second one that
# approaches unity.  Good test of accuracy.
def KMR_Markov_matrix_sequential(N, p, epsilon):
    """
    Generate the Markov matrix for the KMR model with *sequential* move

    N: number of players
    p: level of p-dominance for action 1
       = the value of p such that action 1 is the BR for (1-q, q) for any q > p,
         where q (1-q, resp.) is the prob that the opponent plays action 1 (0, resp.)
    epsilon: mutation probability

    References:
        KMRMarkovMatrixSequential is contributed from https://github.com/oyamad
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


# Tests: methods of MarkovChain, mc_compute_stationary #

def test_markovchain_pmatrices():
    """
    Test the methods of MarkovChain, as well as mc_compute_stationary,
    with P matrix and known solutions
    """
    testset = [
        {'P': np.array([[0.4, 0.6], [0.2, 0.8]]),  # P matrix
         'stationary_dists': np.array([[0.25, 0.75]]),  # Known solution
         'comm_classes': [np.arange(2)],
         'rec_classes': [np.arange(2)],
         'is_irreducible': True,
         'period': 1,
         'is_aperiodic': True,
         'cyclic_classes': [np.arange(2)],
         },
        {'P': np.array([[0, 1], [1, 0]]),
         'stationary_dists': np.array([[0.5, 0.5]]),
         'comm_classes': [np.arange(2)],
         'rec_classes': [np.arange(2)],
         'is_irreducible': True,
         'period': 2,
         'is_aperiodic': False,
         'cyclic_classes': [np.array([0]), np.array([1])],
         },
        {'P': np.eye(2),
         'stationary_dists': np.array([[1, 0], [0, 1]]),
         'comm_classes': [np.array([0]), np.array([1])],
         'rec_classes': [np.array([0]), np.array([1])],
         'is_irreducible': False,
         }
    ]

    # Loop Through TestSet #
    for test_dict in testset:
        mc = MarkovChain(test_dict['P'])
        computed = mc.stationary_distributions
        assert_allclose(computed, test_dict['stationary_dists'])

        assert(mc.num_communication_classes == len(test_dict['comm_classes']))
        assert(mc.is_irreducible == test_dict['is_irreducible'])
        assert(mc.num_recurrent_classes == len(test_dict['rec_classes']))
        assert_array_equal(sorted(mc.communication_classes),
                           sorted(test_dict['comm_classes']))
        assert_array_equal(sorted(mc.recurrent_classes),
                           sorted(test_dict['rec_classes']))
        try:
            assert(mc.period == test_dict['period'])
        except NotImplementedError:
            assert(mc.is_irreducible is False)
        try:
            assert(mc.is_aperiodic == test_dict['is_aperiodic'])
        except NotImplementedError:
            assert(mc.is_irreducible is False)
        try:
            assert_array_equal(sorted(mc.cyclic_classes),
                               sorted(test_dict['cyclic_classes']))
        except NotImplementedError:
            assert(mc.is_irreducible is False)

        # Test of mc_compute_stationary
        computed = mc_compute_stationary(test_dict['P'])
        assert_allclose(computed, test_dict['stationary_dists'])


# Basic Class Structure with Setup #
####################################

class Test_markovchain_stationary_distributions_KMRMarkovMatrix2():
    """
    Test Suite for MarkovChain.stationary_distributions using KMR Markov
    Matrix [suitable for nose]
    """

    # Starting Values #

    N = 27
    epsilon = 1e-2
    p = 1/3
    TOL = 1e-2

    def setUp(self):
        """ Setup a KMRMarkovMatrix and Compute Stationary Values """
        self.P = KMR_Markov_matrix_sequential(self.N, self.p, self.epsilon)
        self.mc = MarkovChain(self.P)
        self.stationary = self.mc.stationary_distributions
        stat_shape = self.stationary.shape

        if len(stat_shape) == 1:
            self.n_stat_dists = 1
        else:
            self.n_stat_dists = stat_shape[0]

    def test_markov_matrix(self):
        "Check that each row of matrix sums to 1"
        mc = self.mc
        assert_allclose(np.sum(mc.P, axis=1), np.ones(mc.n))

    def test_sum_one(self):
        "Check that each stationary distribution sums to 1"
        stationary_distributions = self.stationary
        assert_allclose(np.sum(stationary_distributions, axis=1),
                        np.ones(self.n_stat_dists))

    def test_nonnegative(self):
        "Check that the stationary distributions are non-negative"
        stationary_distributions = self.stationary
        assert(np.all(stationary_distributions > -1e-16))

    def test_left_eigen_vec(self):
        "Check that vP = v for all stationary distributions"
        mc = self.mc
        stationary_distributions = self.stationary

        if self.n_stat_dists == 1:
            assert_allclose(np.dot(stationary_distributions, mc.P),
                            stationary_distributions, atol=self.TOL)
        else:
            for i in range(self.n_stat_dists):
                curr_v = stationary_distributions[i, :]
                assert_allclose(np.dot(curr_v, mc.P), curr_v, atol=self.TOL)


@raises(ValueError)
def test_raises_value_error_non_2dim():
    """Test with non 2dim input"""
    mc = MarkovChain(np.array([0.4, 0.6]))


@raises(ValueError)
def test_raises_value_error_non_sym():
    """Test with non symmetric input"""
    mc = MarkovChain(np.array([[0.4, 0.6]]))


@raises(ValueError)
def test_raises_value_error_non_nonnegative():
    """Test with non nonnegative input"""
    mc = MarkovChain(np.array([[0.4, 0.6], [-0.2, 1.2]]))


@raises(ValueError)
def test_raises_value_error_non_sum_one():
    """Test with input such that some of the rows does not sum to one"""
    mc = MarkovChain(np.array([[0.4, 0.6], [0.2, 0.9]]))


if __name__ == '__main__':
    argv = sys.argv[:]
    argv.append('--verbose')
    argv.append('--nocapture')
    nose.main(argv=argv, defaultTest=__file__)
