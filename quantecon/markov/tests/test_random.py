"""
Tests for markov/random.py

"""
import numpy as np
from numpy.testing import (
    assert_array_equal, assert_raises, assert_array_almost_equal_nulp,
    assert_
)

from quantecon.markov import (
    random_markov_chain, random_stochastic_matrix, random_discrete_dp
)


def test_random_markov_chain_dense():
    sparse = False
    n, k = 5, 3
    mc_dicts = [{'P': random_markov_chain(n, sparse=sparse).P, 'k': n},
                {'P': random_markov_chain(n, k, sparse=sparse).P, 'k': k}]
    for mc_dict in mc_dicts:
        P = mc_dict['P']
        assert_array_equal(P.shape, (n, n))
        assert_array_equal((P > 0).sum(axis=1), [mc_dict['k']]*n)


def test_random_markov_chain_sparse():
    sparse = True
    n, k = 5, 3
    mc_dicts = [{'P': random_markov_chain(n, sparse=sparse).P, 'k': n},
                {'P': random_markov_chain(n, k, sparse=sparse).P, 'k': k}]
    for mc_dict in mc_dicts:
        P = mc_dict['P']
        assert_array_equal(P.shape, (n, n))
        assert_array_equal(P.getnnz(axis=1), [mc_dict['k']]*n)


def test_random_markov_chain_value_error():
    # n <= 0
    assert_raises(ValueError, random_markov_chain, 0)

    # k = 0
    assert_raises(ValueError, random_markov_chain, 2, 0)

    # k > n
    assert_raises(ValueError, random_markov_chain, 2, 3)


def test_random_stochastic_matrix_dense():
    sparse = False
    n, k = 5, 3
    Ps = [random_stochastic_matrix(n, sparse=sparse),
          random_stochastic_matrix(n, k, sparse=sparse)]
    for P in Ps:
        assert_(np.all(P >= 0))
        assert_array_almost_equal_nulp(P.sum(axis=1), np.ones(n))


def test_random_stochastic_matrix_sparse():
    sparse = True
    n, k = 5, 3
    Ps = [random_stochastic_matrix(n, sparse=sparse),
          random_stochastic_matrix(n, k, sparse=sparse)]
    for P in Ps:
        assert_(np.all(P.data >= 0))
        assert_array_almost_equal_nulp(P.sum(axis=1), np.ones(n))


def test_random_stochastic_matrix_dense_vs_sparse():
    n, k = 10, 5
    seed = 1234
    P_dense = random_stochastic_matrix(n, sparse=False, random_state=seed)
    P_sparse = random_stochastic_matrix(n, sparse=True, random_state=seed)
    assert_array_equal(P_dense, P_sparse.toarray())

    P_dense = random_stochastic_matrix(n, k, sparse=False, random_state=seed)
    P_sparse = random_stochastic_matrix(n, k, sparse=True, random_state=seed)
    assert_array_equal(P_dense, P_sparse.toarray())


def test_random_stochastic_matrix_k_1():
    n, k = 3, 1
    P_dense = random_stochastic_matrix(n, k, sparse=False)
    P_sparse = random_stochastic_matrix(n, k, sparse=True)
    assert_array_equal(P_dense[P_dense != 0], np.ones(n))
    assert_array_equal(P_sparse.data, np.ones(n))
    for P in [P_dense, P_sparse]:
        assert_array_almost_equal_nulp(P.sum(axis=1), np.ones(n))


class TestRandomDiscreteDP:
    def setup_method(self):
        self.num_states, self.num_actions = 5, 4
        self.num_sa = self.num_states * self.num_actions
        self.k = 3
        seed = 1234

        self.ddp = \
            random_discrete_dp(self.num_states, self.num_actions, k=self.k,
                               sparse=False, sa_pair=False, random_state=seed)

        labels = ['dense', 'sparse']
        self.ddps_sa = {}
        for label in labels:
            is_sparse = (label == 'sparse')
            self.ddps_sa[label] = \
                random_discrete_dp(self.num_states, self.num_actions, k=self.k,
                                   sparse=is_sparse, sa_pair=True,
                                   random_state=seed)

    def test_shape(self):
        n, m, L = self.num_states, self.num_actions, self.num_sa

        assert_(self.ddp.R.shape == (n, m))
        assert_(self.ddp.Q.shape == (n, m, n))

        for ddp in self.ddps_sa.values():
            assert_(ddp.R.shape == (L,))
            assert_(ddp.Q.shape == (L, n))

    def test_nonzero(self):
        n, m, L, k = self.num_states, self.num_actions, self.num_sa, self.k

        assert_array_equal((self.ddp.Q > 0).sum(axis=-1), np.ones((n, m))*k)
        assert_array_equal((self.ddps_sa['dense'].Q > 0).sum(axis=-1),
                           np.ones(L)*k)
        assert_array_equal(self.ddps_sa['sparse'].Q.getnnz(axis=-1),
                           np.ones(L)*k)

    def test_equal_reward(self):
        assert_array_equal(self.ddp.R.ravel(), self.ddps_sa['dense'].R)
        assert_array_equal(self.ddps_sa['dense'].R, self.ddps_sa['sparse'].R)

    def test_equal_probability(self):
        assert_array_equal(self.ddp.Q.ravel(), self.ddps_sa['dense'].Q.ravel())
        assert_array_equal(self.ddps_sa['dense'].Q,
                           self.ddps_sa['sparse'].Q.toarray())

    def test_equal_beta(self):
        for ddp in self.ddps_sa.values():
            assert_(ddp.beta == self.ddp.beta)
