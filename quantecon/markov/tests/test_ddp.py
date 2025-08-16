"""
Tests for markov/ddp.py

"""
import numpy as np
import scipy.sparse as sparse
from numpy.testing import (assert_array_equal, assert_allclose, assert_raises,
                           assert_)

from quantecon.markov import DiscreteDP, backward_induction


class TestDiscreteDP:
    def setup_method(self):
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

            # Check v is an epsilon/2-approximation of v_star
            assert_(np.abs(res.v - self.v_star).max() < self.epsilon/2)
            assert_(np.abs(res_init.v - self.v_star).max() < self.epsilon/2)

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

            # Check v is an epsilon/2-approximation of v_star
            assert_(np.abs(res.v - self.v_star).max() < self.epsilon/2)
            assert_(np.abs(res_init.v - self.v_star).max() < self.epsilon/2)

            # Check sigma == sigma_star
            assert_array_equal(res.sigma, self.sigma_star)
            assert_array_equal(res_init.sigma, self.sigma_star)

    def test_modified_policy_iteration_k0(self):
        k = 0
        for ddp in self.ddps:
            res = ddp.solve(method='modified_policy_iteration',
                            epsilon=self.epsilon, k=k)

            # Check v is an epsilon/2-approximation of v_star
            assert_(np.abs(res.v - self.v_star).max() < self.epsilon/2)

            # Check sigma == sigma_star
            assert_array_equal(res.sigma, self.sigma_star)

    def test_linear_programming(self):
        for ddp in self.ddps:
            if ddp._sparse:
                assert_raises(NotImplementedError, ddp.solve,
                              method='linear_programming')
            else:
                res = ddp.solve(method='linear_programming')

                v_init = [0, 1]
                res_init = ddp.solve(method='linear_programming',
                                     v_init=v_init)

                # Check v == v_star
                assert_allclose(res.v, self.v_star)
                assert_allclose(res_init.v, self.v_star)

                # Check sigma == sigma_star
                assert_array_equal(res.sigma, self.sigma_star)
                assert_array_equal(res_init.sigma, self.sigma_star)


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
    ddp1 = ddp0.to_sa_pair_form()
    ddp2 = ddp0.to_sa_pair_form(sparse=False)
    methods = ['vi', 'pi', 'mpi', 'lp']

    for ddp in [ddp0, ddp1, ddp2]:
        for method in methods:
            if method == 'lp' and ddp._sparse:
                assert_raises(NotImplementedError, ddp.solve,
                              method=method, v_init=v_init)
            else:
                res = ddp.solve(method=method, v_init=v_init)
                assert_array_equal(res.sigma, sigma_star)
                assert_array_equal(res.v, v_star)


def test_ddp_sorting():
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


class TestFiniteHorizon:
    def setup_method(self):
        # From Puterman 2005, Section 3.2, Section 4.6.1
        # "single-product stochastic inventory control"
        s_indices = [0, 0, 0, 0, 1, 1, 1, 2, 2, 3]
        a_indices = [0, 1, 2, 3, 0, 1, 2, 0, 1, 0]
        R = [ 0., -1., -2., -5.,  5.,  0., -3.,  6., -1.,  5.]
        Q = [[ 1.  ,  0.  ,  0.  ,  0.  ],
             [ 0.75,  0.25,  0.  ,  0.  ],
             [ 0.25,  0.5 ,  0.25,  0.  ],
             [ 0.  ,  0.25,  0.5 ,  0.25],
             [ 0.75,  0.25,  0.  ,  0.  ],
             [ 0.25,  0.5 ,  0.25,  0.  ],
             [ 0.  ,  0.25,  0.5 ,  0.25],
             [ 0.25,  0.5 ,  0.25,  0.  ],
             [ 0.  ,  0.25,  0.5 ,  0.25],
             [ 0.  ,  0.25,  0.5 ,  0.25]]
        beta = 1
        self.ddp = DiscreteDP(R, Q, beta, s_indices, a_indices)

    def test_backward_induction(self):
        T = 3
        # v_T = np.zeors(self.ddp.n)
        vs_expected = [[67/16, 129/16, 194/16, 227/16],
                       [2, 25/4, 10, 21/2],
                       [0, 5, 6, 5],
                       [0, 0, 0, 0]]
        sigmas_expected = [[3, 0, 0, 0],
                           [2, 0, 0, 0],
                           [0, 0, 0, 0]]

        vs, sigmas = backward_induction(self.ddp, T)

        assert_allclose(vs, vs_expected)
        assert_array_equal(sigmas, sigmas_expected)


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
    # No action is feasible at state 1
    s_indices = [0, 0, 2, 2]
    a_indices = [0, 1, 0, 1]
    R = [1, 0, 0, 1]
    Q = [(1/3, 1/3, 1/3) for i in range(4)]
    beta = 0.95

    assert_raises(ValueError, DiscreteDP, R, Q, beta, s_indices, a_indices)


def test_ddp_beta_1_not_implemented_error():
    n, m = 3, 2
    R = np.array([[0, 1], [1, 0], [0, 1]])
    Q = np.empty((n, m, n))
    Q[:] = 1/n
    beta = 1

    ddp0 = DiscreteDP(R, Q, beta)
    ddp1 = ddp0.to_sa_pair_form()
    ddp2 = ddp0.to_sa_pair_form(sparse=False)

    solution_methods = \
        ['value_iteration', 'policy_iteration', 'modified_policy_iteration']

    for ddp in [ddp0, ddp1, ddp2]:
        assert_raises(NotImplementedError, ddp.evaluate_policy, np.zeros(n))
        for method in solution_methods:
            assert_raises(NotImplementedError, getattr(ddp, method))


def test_ddp_to_sa_and_to_product():
    n, m = 3, 2
    R = np.array([[0, 1], [1, 0], [-np.inf, 1]])
    Q = np.empty((n, m, n))
    Q[:] = 1/n
    Q[0, 0, 0] = 0
    Q[0, 0, 1] = 2/n
    beta = 0.95

    sparse_R = np.array([0, 1, 1, 0, 1])
    _Q = np.full((5, 3), 1/3)
    _Q[0, 0] = 0
    _Q[0, 1] = 2/n
    sparse_Q = sparse.coo_matrix(_Q)

    ddp = DiscreteDP(R, Q, beta)
    ddp_sa = ddp.to_sa_pair_form()
    ddp_sa2 = ddp_sa.to_sa_pair_form()
    ddp_sa3 = ddp.to_sa_pair_form(sparse=False)
    ddp2 = ddp_sa.to_product_form()
    ddp3 = ddp_sa2.to_product_form()
    ddp4 = ddp.to_product_form()

    # make sure conversion worked
    for ddp_s in [ddp_sa, ddp_sa2, ddp_sa3]:
        assert_allclose(ddp_s.R, sparse_R)
        # allcose doesn't work on sparse
        np.max(np.abs((sparse_Q - ddp_s.Q))) < 1e-15
        assert_allclose(ddp_s.beta, beta)

    # these two will have probability 0 in state 2, action 0 b/c
    # of the infeasiability in R
    funky_Q = np.empty((n, m, n))
    funky_Q[:] = 1/n
    funky_Q[0, 0, 0] = 0
    funky_Q[0, 0, 1] = 2/n
    funky_Q[2, 0, :] = 0
    for ddp_f in [ddp2, ddp3]:
        assert_allclose(ddp_f.R, ddp.R)
        assert_allclose(ddp_f.Q, funky_Q)
        assert_allclose(ddp_f.beta, ddp.beta)

    # this one is just the original one.
    assert_allclose(ddp4.R, ddp.R)
    assert_allclose(ddp4.Q, ddp.Q)
    assert_allclose(ddp4.beta, ddp.beta)

    for method in ["pi", "vi", "mpi"]:
        sol1 = ddp.solve(method=method)
        for ddp_other in [ddp_sa, ddp_sa2, ddp_sa3, ddp2, ddp3, ddp4]:
            sol2 = ddp_other.solve(method=method)

            for k in ["v", "sigma", "num_iter"]:
                assert_allclose(sol1[k], sol2[k])
