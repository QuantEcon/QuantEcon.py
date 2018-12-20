"""
Tests for bimatrix_generators.py

"""
import numpy as np
from scipy.special import comb
from numpy.testing import assert_array_equal
from nose.tools import eq_, ok_, raises
from quantecon.gridtools import num_compositions
from quantecon.game_theory import pure_nash_brute

from quantecon.game_theory import (
    blotto_game, ranking_game, sgc_game, tournament_game, unit_vector_game,
)


class TestBlottoGame:
    def setUp(self):
        self.h, self.t = 4, 3
        rho = 0.5
        self.g = blotto_game(self.h, self.t, rho)

    def test_size(self):
        n = num_compositions(self.h, self.t)
        eq_(self.g.nums_actions, (n, n))

    def test_constant_diagonal(self):
        for i in range(self.g.N):
            diag = self.g.payoff_arrays[i].diagonal()
            ok_((diag == diag[0]).all())

    def test_seed(self):
        seed = 0
        h, t = 3, 4
        rho = -0.5
        g0 = blotto_game(h, t, rho, random_state=seed)
        g1 = blotto_game(h, t, rho, random_state=seed)
        assert_array_equal(g1.payoff_profile_array, g0.payoff_profile_array)


class TestRankingGame:
    def setUp(self):
        self.n = 100
        self.g = ranking_game(self.n)

    def test_size(self):
        eq_(self.g.nums_actions, (self.n, self.n))

    def test_weakly_decreasing_row_wise_payoffs(self):
        for payoff_array in self.g.payoff_arrays:
            ok_((payoff_array[:, 1:-1] >= payoff_array[:, 2:]).all())

    def test_elements_first_row(self):
        eq_(self.g.payoff_arrays[0][0, 0] + self.g.payoff_arrays[1][0, 0], 1.)
        for payoff_array in self.g.payoff_arrays:
            ok_(np.isin(payoff_array[0, :], [0, 1, 0.5]).all())

    def test_seed(self):
        seed = 0
        n = 100
        g0 = ranking_game(n, random_state=seed)
        g1 = ranking_game(n, random_state=seed)
        assert_array_equal(g1.payoff_profile_array, g0.payoff_profile_array)


def test_sgc_game():
    k = 2
    s = """\
        0.750 0.750 1.000 0.500 0.500 1.000 0.000 0.500 0.000 0.500 0.000 0.500
        0.000 0.500 0.500 1.000 0.750 0.750 1.000 0.500 0.000 0.500 0.000 0.500
        0.000 0.500 0.000 0.500 1.000 0.500 0.500 1.000 0.750 0.750 0.000 0.500
        0.000 0.500 0.000 0.500 0.000 0.500 0.500 0.000 0.500 0.000 0.500 0.000
        0.750 0.000 0.000 0.750 0.000 0.000 0.000 0.000 0.500 0.000 0.500 0.000
        0.500 0.000 0.000 0.750 0.750 0.000 0.000 0.000 0.000 0.000 0.500 0.000
        0.500 0.000 0.500 0.000 0.000 0.000 0.000 0.000 0.750 0.000 0.000 0.750
        0.500 0.000 0.500 0.000 0.500 0.000 0.000 0.000 0.000 0.000 0.000 0.750
        0.750 0.000"""
    bimatrix = np.fromstring(s, sep=' ')
    bimatrix.shape = (4*k-1, 4*k-1, 2)
    bimatrix = bimatrix.swapaxes(0, 1)

    g = sgc_game(k)
    assert_array_equal(g.payoff_profile_array, bimatrix)


class TestTournamentGame:
    def setUp(self):
        self.n = 5
        self.k = 3
        self.m = comb(self.n, self.k, exact=True)
        self.g = tournament_game(self.n, self.k)

    def test_size(self):
        eq_(self.g.nums_actions, (self.n, self.m))

    def test_payoff_values(self):
        possible_values = [0, 1]
        for payoff_array in self.g.payoff_arrays:
            ok_(np.isin(payoff_array, possible_values).all())

        max_num_dominated_subsets = \
            sum([comb(i, self.k, exact=True) for i in range(self.n)])
        ok_(self.g.payoff_arrays[0].sum() <= max_num_dominated_subsets)
        ok_((self.g.payoff_arrays[1].sum(axis=1) == self.k).all())

    def test_seed(self):
        seed = 0
        g0 = tournament_game(self.n, self.k, random_state=seed)
        g1 = tournament_game(self.n, self.k, random_state=seed)
        assert_array_equal(g1.payoff_profile_array, g0.payoff_profile_array)

    @raises(ValueError)
    def test_raises_value_error_too_large_inputs(self):
        n, k = 100, 50
        tournament_game(n, k)


class TestUnitVectorGame:
    def setUp(self):
        self.n = 100
        self.g = unit_vector_game(self.n)

    def test_size(self):
        eq_(self.g.nums_actions, (self.n, self.n))

    def test_payoff_values(self):
        # Player 0
        ok_((np.sum(self.g.players[0].payoff_array, axis=0) == 1).all())

    def test_avoid_pure_nash(self):
        NEs = pure_nash_brute(unit_vector_game(self.n, avoid_pure_nash=True),
                              tol=0)
        eq_(len(NEs), 0)

    def test_seed(self):
        seed = 0
        n = 100
        g0 = unit_vector_game(n, random_state=seed)
        g1 = unit_vector_game(n, random_state=seed)
        assert_array_equal(g1.payoff_profile_array, g0.payoff_profile_array)

    def test_redraw(self):
        seed = 1
        n = 2
        g = unit_vector_game(n, avoid_pure_nash=True, random_state=seed)
        NEs = pure_nash_brute(g, tol=0)
        eq_(len(NEs), 0)

    @raises(ValueError)
    def test_raises_value_error_avoid_pure_nash_n_1(self):
        n = 1
        unit_vector_game(n, avoid_pure_nash=True)


def test_payoff_range():
    func_kwargs_tuples = \
        [(sgc_game, {'k': 3}),
         (tournament_game, {'n': 6, 'k': 3}),
         (unit_vector_game, {'n': 5}),
         (unit_vector_game, {'n': 5, 'avoid_pure_nash': True})]
    for func, kwargs in func_kwargs_tuples:
        g = func(**kwargs)
        for payoff_array in g.payoff_arrays:
            assert_array_equal(np.clip(payoff_array, 0, 1), payoff_array)


if __name__ == '__main__':
    import sys
    import nose

    argv = sys.argv[:]
    argv.append('--verbose')
    argv.append('--nocapture')
    nose.main(argv=argv, defaultTest=__file__)
