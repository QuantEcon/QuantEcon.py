"""
Filename: test_random_mc.py
Author: Daisuke Oyama

Tests for random_mc.py

"""
import numpy as np
from numpy.testing import assert_array_equal, assert_raises
from nose.tools import eq_, ok_, raises

from quantecon.random_mc import (
    random_markov_chain, random_stochastic_matrix,
    random_choice_without_replacement
)


def test_random_markov_chain_dense():
    n, k = 10, 5
    mc = random_markov_chain(n, k, sparse=False)
    assert_array_equal(mc.P.shape, (n, n))


@raises(NotImplementedError)
def test_random_markov_chain_sparse():
    n, k = 10, 5
    mc = random_markov_chain(n, k, sparse=True)
    assert_array_equal(mc.P.shape, (n, n))
    eq_(mc.P.nnz, n*k)


def test_random_markov_chain_value_error():
    # n <= 0
    assert_raises(ValueError, random_markov_chain, 0)

    # k = 0
    assert_raises(ValueError, random_markov_chain, 2, 0)

    # k > n
    assert_raises(ValueError, random_markov_chain, 2, 3)


def test_random_stochastic_matrix_dense_vs_sparse():
    n, k = 10, 5
    seed = 1234
    P_dense = random_stochastic_matrix(n, sparse=False, random_state=seed)
    P_sparse = random_stochastic_matrix(n, sparse=True, random_state=seed)
    assert_array_equal(P_dense, P_sparse.toarray())

    P_dense = random_stochastic_matrix(n, k, sparse=False, random_state=seed)
    P_sparse = random_stochastic_matrix(n, k, sparse=True, random_state=seed)
    assert_array_equal(P_dense, P_sparse.toarray())


# random_choice_without_replacement #

def test_random_choice_without_replacement_shape():
    assert_array_equal(random_choice_without_replacement(2, 0).shape, (0,))

    n, k, m = 5, 3, 4
    assert_array_equal(
        random_choice_without_replacement(n, k).shape,
        (k,)
    )
    assert_array_equal(
        random_choice_without_replacement(n, k, num_trials=m).shape,
        (m, k)
    )


def test_random_choice_without_replacement_uniqueness():
    n = 10
    a = random_choice_without_replacement(n, n)
    b = np.unique(a)
    eq_(len(b), n)


def test_random_choice_without_replacement_value_error():
    # n <= 0
    assert_raises(ValueError, random_choice_without_replacement, 0, 2)
    assert_raises(ValueError, random_choice_without_replacement, -1, -1)

    # k > n
    assert_raises(ValueError, random_choice_without_replacement, 2, 3)


if __name__ == '__main__':
    import sys
    import nose

    argv = sys.argv[:]
    argv.append('--verbose')
    argv.append('--nocapture')
    nose.main(argv=argv, defaultTest=__file__)
