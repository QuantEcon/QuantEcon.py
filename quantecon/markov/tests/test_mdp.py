"""
Filename: test_mdp.py
Author: Daisuke Oyama

Tests for markov/mdp.py

"""
from __future__ import division

import numpy as np
import scipy.sparse as sparse
from numpy.testing import assert_array_equal, assert_allclose, assert_raises
from nose.tools import eq_, ok_

from quantecon.markov import MDP


class TestMDP:
    def setUp(self):
        # From Puterman 2005, Section 3.1
        beta = 0.95

        # Formulation with R: n x m, Q: n x m x n
        n, m = 2, 2  # number of states, number of actions
        R = [[5, 10], [-1, -np.inf]]
        Q = np.empty((n, m, n))
        Q[0, 0, :] = 0.5, 0.5
        Q[0, 1, :] = 0, 1
        Q[1, :, :] = 0, 1
        mdp0 = MDP(R, Q, beta)

        # Formulation with state-action pairs
        L = 3  # number of state-action pairs
        s_indices = [0, 0, 1]
        a_indices = [0, 1, 0]
        R_sa = [R[0][0], R[0][1], R[1][0]]
        Q_sa = sparse.lil_matrix((L, n))
        Q_sa[0, :] = Q[0, 0, :]
        Q_sa[1, :] = Q[0, 1, :]
        Q_sa[2, :] = Q[1, 0, :]
        mdp_sa_sparse = MDP(R_sa, Q_sa, beta, s_indices, a_indices)
        mdp_sa_dense = MDP(R_sa, Q_sa.toarray(), beta, s_indices, a_indices)

        self.mdps = [mdp0, mdp_sa_sparse, mdp_sa_dense]

        for mdp in self.mdps:
            mdp.max_iter = 200

        self.epsilon = 1e-2

        # Analytical solution for beta > 10/11, Example 6.2.1
        self.v_star = [(5-5.5*beta)/((1-0.5*beta)*(1-beta)), -1/(1-beta)]
        self.sigma_star = [0, 0]

    def test_value_iteration(self):
        for mdp in self.mdps:
            res = mdp.solve(method='value_iteration', epsilon=self.epsilon)

            v_init = [0, 0]
            res_init = mdp.solve(method='value_iteration', v_init=v_init,
                                 epsilon=self.epsilon)

            # Check v is an epsilon/2-approxmation of v_star
            ok_(np.abs(res.v - self.v_star).max() < self.epsilon/2)
            ok_(np.abs(res_init.v - self.v_star).max() < self.epsilon/2)

            # Check sigma == sigma_star
            assert_array_equal(res.sigma, self.sigma_star)
            assert_array_equal(res_init.sigma, self.sigma_star)

    def test_policy_iteration(self):
        for mdp in self.mdps:
            res = mdp.solve(method='policy_iteration')

            v_init = [0, 1]
            res_init = mdp.solve(method='policy_iteration', v_init=v_init)

            # Check v == v_star
            assert_allclose(res.v, self.v_star)
            assert_allclose(res_init.v, self.v_star)

            # Check sigma == sigma_star
            assert_array_equal(res.sigma, self.sigma_star)
            assert_array_equal(res_init.sigma, self.sigma_star)

    def test_modified_policy_iteration(self):
        for mdp in self.mdps:
            res = mdp.solve(method='modified_policy_iteration',
                            epsilon=self.epsilon)

            v_init = [0, 1]
            res_init = mdp.solve(method='modified_policy_iteration',
                                 v_init=v_init,
                                 epsilon=self.epsilon)

            # Check v is an epsilon/2-approxmation of v_star
            ok_(np.abs(res.v - self.v_star).max() < self.epsilon/2)
            ok_(np.abs(res_init.v - self.v_star).max() < self.epsilon/2)

            # Check sigma == sigma_star
            assert_array_equal(res.sigma, self.sigma_star)
            assert_array_equal(res_init.sigma, self.sigma_star)

    def test_modified_policy_iteration_k0(self):
        k = 0
        for mdp in self.mdps:
            res = mdp.solve(method='modified_policy_iteration',
                            epsilon=self.epsilon, k=k)

            # Check v is an epsilon/2-approxmation of v_star
            ok_(np.abs(res.v - self.v_star).max() < self.epsilon/2)

            # Check sigma == sigma_star
            assert_array_equal(res.sigma, self.sigma_star)


def test_mdp_beta_0():
    n, m = 3, 2
    R = np.array([[0, 1], [1, 0], [0, 1]])
    Q = np.empty((n, m, n))
    Q[:] = 1/n
    beta = 0
    sigma_star = [1, 0, 1]
    v_star = [1, 1, 1]
    v_init = [0, 0, 0]

    mdp0 = MDP(R, Q, beta)
    s_indices, a_indices = np.where(R > -np.inf)
    R_sa = R[s_indices, a_indices]
    Q_sa = Q[s_indices, a_indices]
    mdp1 = MDP(R_sa, Q_sa, beta, s_indices, a_indices)
    methods = ['vi', 'pi', 'mpi']

    for mdp in [mdp0, mdp1]:
        for method in methods:
            res = mdp.solve(method=method, v_init=v_init)
            assert_array_equal(res.sigma, sigma_star)
            assert_array_equal(res.v, v_star)


def test_mdp_negative_inf_error():
    n, m = 3, 2
    R = np.array([[0, 1], [0, -np.inf], [-np.inf, -np.inf]])
    Q = np.empty((n, m, n))
    Q[:] = 1/n

    s_indices = [0, 0, 1, 1, 2, 2]
    a_indices = [0, 1, 0, 1, 0, 1]
    R_sa = R.reshape(n*m)
    Q_sa_dense = Q.reshape(n*m, n)
    Q_sa_sparse = sparse.csr_matrix(Q_sa_dense)

    beta = 0.95

    assert_raises(ValueError, MDP, R, Q, beta)
    assert_raises(
        ValueError, MDP, R_sa, Q_sa_dense, beta, s_indices, a_indices
    )
    assert_raises(
        ValueError, MDP, R_sa, Q_sa_sparse, beta, s_indices, a_indices
    )


def test_mdp_no_feasibile_action_error():
    n, m = 3, 2

    # No action is feasible at state 1
    s_indices = [0, 0, 2, 2]
    a_indices = [0, 1, 0, 1]
    R = [1, 0, 0, 1]
    Q = [(1/2, 1/2) for i in range(4)]
    beta = 0.95

    assert_raises(ValueError, MDP, R, Q, beta, s_indices, a_indices)


if __name__ == '__main__':
    import sys
    import nose

    argv = sys.argv[:]
    argv.append('--verbose')
    argv.append('--nocapture')
    nose.main(argv=argv, defaultTest=__file__)
