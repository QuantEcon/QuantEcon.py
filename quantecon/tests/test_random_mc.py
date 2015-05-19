"""
Filename: test_random_mc.py
Author: Daisuke Oyama

Tests for random_mc.py

"""
import numpy as np
from numpy.testing import assert_array_equal
from nose.tools import eq_, ok_, raises

from quantecon.random_mc import random_markov_chain, random_stochastic_matrix


def test_random_stochastic_matrix_dense_vs_sparse():
    n, k = 10, 5
    seed = 1234
    P_dense = random_stochastic_matrix(n, k, sparse=False, random_state=seed)
    P_sparse = random_stochastic_matrix(n, k, sparse=True, random_state=seed)
    assert_array_equal(P_dense, P_sparse.toarray())


@raises(NotImplementedError)
def test_random_markov_chain_sparse():
    n, k = 10, 5
    random_markov_chain(n, k, sparse=True)


if __name__ == '__main__':
    import sys
    import nose

    argv = sys.argv[:]
    argv.append('--verbose')
    argv.append('--nocapture')
    nose.main(argv=argv, defaultTest=__file__)
