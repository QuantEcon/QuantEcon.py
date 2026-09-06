"""
Tests for lemke_howson.py
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_, assert_raises
from quantecon.game_theory import Player, NormalFormGame, lemke_howson
from quantecon.game_theory.lemke_howson import (
    _initialize_tableaux, _lemke_howson_tbl
)


class TestLemkeHowson():
    def setup_method(self):
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
    def setup_method(self):
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
                assert_(res.converged == d['converged'])


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
        assert_(res0.init == res1.init)

    init_pivot = 1
    max_iter = m+n
    NE, res = lemke_howson(g, init_pivot=init_pivot, max_iter=max_iter,
                           capping=1, full_output=True)
    assert_(res.num_iter == max_iter)
    assert_(res.init == init_pivot-1)


def test_lemke_howson_invalid_g():
    bimatrix = [[(3, 3), (3, 2)],
                [(2, 2), (5, 6)],
                [(0, 3), (6, 1)]]
    assert_raises(TypeError, lemke_howson, bimatrix)


def test_lemke_howson_invalid_init_pivot_integer():
    bimatrix = [[(3, 3), (3, 2)],
                [(2, 2), (5, 6)],
                [(0, 3), (6, 1)]]
    g = NormalFormGame(bimatrix)
    assert_raises(ValueError, lemke_howson, g, -1)


def test_lemke_howson_invalid_init_pivot_float():
    bimatrix = [[(3, 3), (3, 2)],
                [(2, 2), (5, 6)],
                [(0, 3), (6, 1)]]
    g = NormalFormGame(bimatrix)
    assert_raises(TypeError, lemke_howson, g, 1.0)


@pytest.mark.xfail(
    strict=True,
    reason="`tol_ratio_diff` in the lexico-minimum ratio test is absolute, "
           "so with payoffs of order 1e16 the ratios all tie",
)
def test_lemke_howson_scaled_game_returns_nash():
    scale = 1e16
    A = scale * np.array([[3., 1.],
                          [1., 3.]])
    B = scale * np.array([[1., 3.],
                          [3., 1.]])
    g = NormalFormGame((A, B))
    NE, res = lemke_howson(g, full_output=True)
    assert_(res.converged)
    assert_(g.is_nash(NE))


def test_lemke_howson_tbl_breakdown():
    # No positive entry in the column of the initial pivot: the routine
    # must stop and report non-convergence rather than pivot on a
    # meaningless row
    A = np.array([[3, 3], [2, 5], [0, 6]])
    B = np.array([[3, 2, 3], [2, 6, 1]])
    m, n = A.shape
    tableaux = (np.empty((n, m+n+1)), np.empty((m, m+n+1)))
    bases = (np.empty(n, dtype=int), np.empty(m, dtype=int))
    _initialize_tableaux((A, B), tableaux, bases)
    init_pivot = 0
    tableaux[0][:, init_pivot] = 0
    converged, num_iter = _lemke_howson_tbl(tableaux, bases, init_pivot, 10)
    assert_(not converged)
    assert_(num_iter == 0)
