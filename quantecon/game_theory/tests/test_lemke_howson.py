"""
Tests for lemke_howson.py

"""
import numpy as np
from numpy.testing import assert_allclose
from nose.tools import eq_, raises
from quantecon.game_theory import Player, NormalFormGame, lemke_howson


class TestLemkeHowson():
    def setUp(self):
        self.game_dicts = []

        # From von Stengel 2007 in Algorithmic Game Theory
        bimatrix = [[(3, 3), (3, 2)],
                    [(2, 2), (5, 6)],
                    [(0, 3), (6, 1)]]
        NEs_dict = {0: ([1, 0, 0], [1, 0]),
                    1: ([0, 1/3, 2/3], [1/3, 2/3])}  # init_pivot: NE
        d = {'g': NormalFormGame(bimatrix),
             'NEs_dict': NEs_dict}
        self.game_dicts.append(d)

    def test_lemke_howson(self):
        for d in self.game_dicts:
            for k in d['NEs_dict'].keys():
                NE_computed = lemke_howson(d['g'], init_pivot=k)
                for action_computed, action in zip(NE_computed,
                                                   d['NEs_dict'][k]):
                    assert_allclose(action_computed, action)


class TestLemkeHowsonDegenerate():
    def setUp(self):
        self.game_dicts = []

        # From von Stengel 2007 in Algorithmic Game Theory
        bimatrix = [[(3, 3), (3, 3)],
                    [(2, 2), (5, 6)],
                    [(0, 3), (6, 1)]]
        NEs_dict = {0: ([0, 1/3, 2/3], [1/3, 2/3])}
        d = {'g': NormalFormGame(bimatrix),
             'NEs_dict': NEs_dict,
             'converged': True}
        self.game_dicts.append(d)

        # == Examples of cycles by "ad hoc" tie breaking rules == #

        # Example where tie breaking that picks the variable with
        # the smallest row index in the tableau leads to cycling
        A = np.array([[0, 0, 0],
                      [0, 1, 1],
                      [1, 1, 0]])
        B = np.array([[1, 0, 1],
                      [1, 1, 0],
                      [0, 0, 2]])
        NEs_dict = {0: ([0, 2/3, 1/3], [0, 1, 0])}
        d = {'g': NormalFormGame((Player(A), Player(B))),
             'NEs_dict': NEs_dict,
             'converged': True}
        self.game_dicts.append(d)

        # Example where tie breaking that picks the variable with
        # the smallest variable index in the tableau leads to cycling
        perm = [2, 0, 1]
        C = A[:, perm]
        D = B[perm, :]
        NEs_dict = {0: ([0, 2/3, 1/3], [0, 0, 1])}
        d = {'g': NormalFormGame((Player(C), Player(D))),
             'NEs_dict': NEs_dict,
             'converged': True}
        self.game_dicts.append(d)

    def test_lemke_howson_degenerate(self):
        for d in self.game_dicts:
            for k in d['NEs_dict'].keys():
                NE_computed, res = lemke_howson(d['g'], init_pivot=k,
                                                full_output=True)
                for action_computed, action in zip(NE_computed,
                                                   d['NEs_dict'][k]):
                    assert_allclose(action_computed, action)
                eq_(res.converged, d['converged'])


def test_lemke_howson_capping():
    bimatrix = [[(3, 3), (3, 2)],
                [(2, 2), (5, 6)],
                [(0, 3), (6, 1)]]
    g = NormalFormGame(bimatrix)
    m, n = g.nums_actions
    max_iter = 10**6  # big number

    for k in range(m+n):
        NE0, res0 = lemke_howson(g, init_pivot=k, max_iter=max_iter,
                                 capping=None, full_output=True)
        NE1, res1 = lemke_howson(g, init_pivot=k, max_iter=max_iter,
                                 capping=max_iter, full_output=True)
        for action0, action1 in zip(NE0, NE1):
            assert_allclose(action0, action1)
        eq_(res0.init, res1.init)

    init_pivot = 1
    max_iter = m+n
    NE, res = lemke_howson(g, init_pivot=init_pivot, max_iter=max_iter,
                           capping=1, full_output=True)
    eq_(res.num_iter, max_iter)
    eq_(res.init, init_pivot-1)


@raises(TypeError)
def test_lemke_howson_invalid_g():
    bimatrix = [[(3, 3), (3, 2)],
                [(2, 2), (5, 6)],
                [(0, 3), (6, 1)]]
    lemke_howson(bimatrix)


@raises(ValueError)
def test_lemke_howson_invalid_init_pivot_integer():
    bimatrix = [[(3, 3), (3, 2)],
                [(2, 2), (5, 6)],
                [(0, 3), (6, 1)]]
    g = NormalFormGame(bimatrix)
    lemke_howson(g, -1)


@raises(TypeError)
def test_lemke_howson_invalid_init_pivot_float():
    bimatrix = [[(3, 3), (3, 2)],
                [(2, 2), (5, 6)],
                [(0, 3), (6, 1)]]
    g = NormalFormGame(bimatrix)
    lemke_howson(g, 1.0)


if __name__ == '__main__':
    import sys
    import nose

    argv = sys.argv[:]
    argv.append('--verbose')
    argv.append('--nocapture')
    nose.main(argv=argv, defaultTest=__file__)
