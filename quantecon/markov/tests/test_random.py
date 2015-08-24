"""
Filename: test_random.py
Author: Daisuke Oyama

Tests for markov/random.py

"""
import numpy as np
from numpy.testing import (
    assert_array_equal, assert_raises, assert_array_almost_equal_nulp
)
from nose.tools import eq_, ok_, raises

from quantecon.markov import (
    random_markov_chain, random_stochastic_matrix, random_mdp
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
        ok_(np.all(P >= 0))
        assert_array_almost_equal_nulp(P.sum(axis=1), np.ones(n))


def test_random_stochastic_matrix_sparse():
    sparse = True
    n, k = 5, 3
    Ps = [random_stochastic_matrix(n, sparse=sparse),
          random_stochastic_matrix(n, k, sparse=sparse)]
    for P in Ps:
        ok_(np.all(P.data >= 0))
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


class TestRandomMDP:
    def setUp(self):
        self.num_states, self.num_actions = 5, 4
        self.num_sa = self.num_states * self.num_actions
        self.k = 3
        seed = 1234

        self.mdp = random_mdp(self.num_states, self.num_actions, k=self.k,
                              sparse=False, sa_pair=False, random_state=seed)

        labels = ['dense', 'sparse']
        self.mdps_sa = {}
        for label in labels:
            is_sparse = (label == 'sparse')
            self.mdps_sa[label] = \
                random_mdp(self.num_states, self.num_actions, k=self.k,
                           sparse=is_sparse, sa_pair=True, random_state=seed)

    def test_shape(self):
        n, m, L = self.num_states, self.num_actions, self.num_sa

        eq_(self.mdp.R.shape, (n, m))
        eq_(self.mdp.Q.shape, (n, m, n))

        for mdp in self.mdps_sa.values():
            eq_(mdp.R.shape, (L,))
            eq_(mdp.Q.shape, (L, n))

    def test_nonzero(self):
        n, m, L, k = self.num_states, self.num_actions, self.num_sa, self.k

        assert_array_equal((self.mdp.Q > 0).sum(axis=-1), np.ones((n, m))*k)
        assert_array_equal((self.mdps_sa['dense'].Q > 0).sum(axis=-1),
                           np.ones(L)*k)
        assert_array_equal(self.mdps_sa['sparse'].Q.getnnz(axis=-1),
                           np.ones(L)*k)

    def test_equal_reward(self):
        assert_array_equal(self.mdp.R.ravel(), self.mdps_sa['dense'].R)
        assert_array_equal(self.mdps_sa['dense'].R, self.mdps_sa['sparse'].R)

    def test_equal_probability(self):
        assert_array_equal(self.mdp.Q.ravel(), self.mdps_sa['dense'].Q.ravel())
        assert_array_equal(self.mdps_sa['dense'].Q,
                           self.mdps_sa['sparse'].Q.toarray())

    def test_equal_beta(self):
        for mdp in self.mdps_sa.values():
            eq_(mdp.beta, self.mdp.beta)


if __name__ == '__main__':
    import sys
    import nose

    argv = sys.argv[:]
    argv.append('--verbose')
    argv.append('--nocapture')
    nose.main(argv=argv, defaultTest=__file__)
