"""
Tests for mc_tools.py

Functions
---------
    mc_compute_stationary   [Status: 1 x Simple Test Written]
    mc_sample_path          [Status: TBD]

"""

from __future__ import division

import numpy as np
import unittest
from numpy.testing import assert_allclose

from quantecon.mc_tools import DMarkov, mc_compute_stationary, mc_sample_path

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

### Tests: mc_compute_stationary ###

def test_mc_compute_stationary_pmatrices():
    """
        Test mc_compute_stationary with P Matrix and Known Solutions
    """

                    #-P Matrix-#                        , #-Known Solution-#
    testset =   [
                    ( np.array([[0.4,0.6], [0.2,0.8]]), np.array([0.25, 0.75]) ),
                    ( np.eye(2), np.eye(2) )
                ]

    #-Loop Through TestSet-#
    for (P, known) in testset:
        computed = mc_compute_stationary(P)
        assert_allclose(computed, known)





# Basic Class Structure with Setup #
####################################

class Test_mc_compute_stationary_KMRMarkovMatrix2():
    """
    Test Suite for mc_compute_stationary using KMR Markov Matrix [suitable for nose]
    """

    #-Starting Values-#

    N = 27
    epsilon = 1e-2
    p = 1/3
    TOL = 1e-2

    def setUp(self):
        """ Setup a KMRMarkovMatrix and Compute Stationary Values """
        self.P = KMR_Markov_matrix_sequential(self.N, self.p, self.epsilon)
        self.mc = DMarkov(self.P)
        self.stationary = self.mc.mc_compute_stationary()
        stat_shape = self.stationary.shape

        if len(stat_shape) is 1:
            self.n_stat_dists = 1
        else:
            self.n_stat_dists = stat_shape[1]


    def test_markov_matrix(self):
        "Check that each row of matrix sums to 1"
        mc = self.mc
        assert_allclose(np.sum(mc.P, axis=1), np.ones(mc.n))

    def test_sum_one(self):
        "Check each stationary distribution sums to 1"
        stationary_distributions = self.stationary
        assert_allclose(np.sum(stationary_distributions, axis=0),
                        np.ones(self.n_stat_dists))

    def test_nonnegative(self):
        "Check that the stationary distributions are non-negative"
        stationary_distributions = self.stationary
        assert(np.all(stationary_distributions > -1e-16))

    def test_left_eigen_vec(self):
        "Check that vP = v for all stationary distributions"
        mc = self.mc
        stationary = self.stationary

        if self.n_stat_dists is 1:
            assert_allclose(np.dot(stationary, mc.P), stationary, atol=self.TOL)
        else:
            for i in range(self.n_stat_dists):
                curr_v = stationary_distributions[:, i]
                assert_allclose(np.dot(curr_v, mc.P), curr_v, atol=self.TOL)


