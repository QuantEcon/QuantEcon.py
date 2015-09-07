"""
Filename: test_ddp.py
Author: Daisuke Oyama

Tests for markov/ddp.py

"""
from __future__ import division

import numpy as np
import scipy.sparse as sparse
from numpy.testing import assert_array_equal, assert_allclose, assert_raises
from nose.tools import eq_, ok_

from quantecon.markov import DiscreteDP


class TestDiscreteDP:
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
        ddp0 = DiscreteDP(R, Q, beta)

        # Formulation with state-action pairs
        L = 3  # number of state-action pairs
        s_indices = [0, 0, 1]
        a_indices = [0, 1, 0]
        R_sa = [R[0][0], R[0][1], R[1][0]]
        Q_sa = sparse.lil_matrix((L, n))
        Q_sa[0, :] = Q[0, 0, :]
        Q_sa[1, :] = Q[0, 1, :]
        Q_sa[2, :] = Q[1, 0, :]
        ddp_sa_sparse = DiscreteDP(R_sa, Q_sa, beta, s_indices, a_indices)
        ddp_sa_dense = \
            DiscreteDP(R_sa, Q_sa.toarray(), beta, s_indices, a_indices)

        self.ddps = [ddp0, ddp_sa_sparse, ddp_sa_dense]

        for ddp in self.ddps:
            ddp.max_iter = 200

        self.epsilon = 1e-2

        # Analytical solution for beta > 10/11, Example 6.2.1
        self.v_star = [(5-5.5*beta)/((1-0.5*beta)*(1-beta)), -1/(1-beta)]
        self.sigma_star = [0, 0]

    def test_value_iteration(self):
        for ddp in self.ddps:
            res = ddp.solve(method='value_iteration', epsilon=self.epsilon)

            v_init = [0, 0]
            res_init = ddp.solve(method='value_iteration', v_init=v_init,
                                 epsilon=self.epsilon)

            # Check v is an epsilon/2-approxmation of v_star
            ok_(np.abs(res.v - self.v_star).max() < self.epsilon/2)
            ok_(np.abs(res_init.v - self.v_star).max() < self.epsilon/2)

            # Check sigma == sigma_star
            assert_array_equal(res.sigma, self.sigma_star)
            assert_array_equal(res_init.sigma, self.sigma_star)

    def test_policy_iteration(self):
        for ddp in self.ddps:
            res = ddp.solve(method='policy_iteration')

            v_init = [0, 1]
            res_init = ddp.solve(method='policy_iteration', v_init=v_init)

            # Check v == v_star
            assert_allclose(res.v, self.v_star)
            assert_allclose(res_init.v, self.v_star)

            # Check sigma == sigma_star
            assert_array_equal(res.sigma, self.sigma_star)
            assert_array_equal(res_init.sigma, self.sigma_star)

    def test_modified_policy_iteration(self):
        for ddp in self.ddps:
            res = ddp.solve(method='modified_policy_iteration',
                            epsilon=self.epsilon)

            v_init = [0, 1]
            res_init = ddp.solve(method='modified_policy_iteration',
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
        for ddp in self.ddps:
            res = ddp.solve(method='modified_policy_iteration',
                            epsilon=self.epsilon, k=k)

            # Check v is an epsilon/2-approxmation of v_star
            ok_(np.abs(res.v - self.v_star).max() < self.epsilon/2)

            # Check sigma == sigma_star
            assert_array_equal(res.sigma, self.sigma_star)


def test_ddp_beta_0():
    n, m = 3, 2
    R = np.array([[0, 1], [1, 0], [0, 1]])
    Q = np.empty((n, m, n))
    Q[:] = 1/n
    beta = 0
    sigma_star = [1, 0, 1]
    v_star = [1, 1, 1]
    v_init = [0, 0, 0]

    ddp0 = DiscreteDP(R, Q, beta)
    s_indices, a_indices = np.where(R > -np.inf)
    R_sa = R[s_indices, a_indices]
    Q_sa = Q[s_indices, a_indices]
    ddp1 = DiscreteDP(R_sa, Q_sa, beta, s_indices, a_indices)
    methods = ['vi', 'pi', 'mpi']

    for ddp in [ddp0, ddp1]:
        for method in methods:
            res = ddp.solve(method=method, v_init=v_init)
            assert_array_equal(res.sigma, sigma_star)
            assert_array_equal(res.v, v_star)


def test_ddp_sorting():
    n, m = 2, 2
    beta = 0.95

    # Sorted
    s_indices = [0, 0, 1]
    a_indices = [0, 1, 0]
    a_indptr = [0, 2, 3]
    R = [0, 1, 2]
    Q = [(1, 0), (1/2, 1/2), (0, 1)]
    Q_sparse = sparse.csr_matrix(Q)

    # Shuffled
    s_indices_shuffled = [0, 1, 0]
    a_indices_shuffled = [0, 0, 1]
    R_shuffled = [0, 2, 1]
    Q_shuffled = [(1, 0), (0, 1), (1/2, 1/2)]
    Q_shuffled_sparse = sparse.csr_matrix(Q_shuffled)

    ddp0 = DiscreteDP(R, Q, beta, s_indices, a_indices)
    ddp_sparse = DiscreteDP(R, Q_sparse, beta, s_indices, a_indices)
    ddp_shuffled = DiscreteDP(R_shuffled, Q_shuffled, beta,
                              s_indices_shuffled, a_indices_shuffled)
    ddp_shuffled_sparse = DiscreteDP(R_shuffled, Q_shuffled_sparse, beta,
                                     s_indices_shuffled, a_indices_shuffled)

    for ddp in [ddp0, ddp_sparse, ddp_shuffled, ddp_shuffled_sparse]:
        assert_array_equal(ddp.s_indices, s_indices)
        assert_array_equal(ddp.a_indices, a_indices)
        assert_array_equal(ddp.a_indptr, a_indptr)
        assert_array_equal(ddp.R, R)
        if sparse.issparse(ddp.Q):
            ddp_Q = ddp.Q.toarray()
        else:
            ddp_Q = ddp.Q
        assert_array_equal(ddp_Q, Q)


def test_ddp_negative_inf_error():
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

    assert_raises(ValueError, DiscreteDP, R, Q, beta)
    assert_raises(
        ValueError, DiscreteDP, R_sa, Q_sa_dense, beta, s_indices, a_indices
    )
    assert_raises(
        ValueError, DiscreteDP, R_sa, Q_sa_sparse, beta, s_indices, a_indices
    )


def test_ddp_no_feasibile_action_error():
    n, m = 3, 2

    # No action is feasible at state 1
    s_indices = [0, 0, 2, 2]
    a_indices = [0, 1, 0, 1]
    R = [1, 0, 0, 1]
    Q = [(1/3, 1/3, 1/3) for i in range(4)]
    beta = 0.95

    assert_raises(ValueError, DiscreteDP, R, Q, beta, s_indices, a_indices)


if __name__ == '__main__':
    import sys
    import nose

    argv = sys.argv[:]
    argv.append('--verbose')
    argv.append('--nocapture')
    nose.main(argv=argv, defaultTest=__file__)
