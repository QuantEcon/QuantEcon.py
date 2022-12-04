"""
Tests for mclennan_tourky.py
"""
import numpy as np
from numpy.testing import assert_array_equal, assert_raises, assert_
from quantecon.game_theory import Player, NormalFormGame, mclennan_tourky
from quantecon.game_theory.mclennan_tourky import (
    _best_response_selection, _flatten_action_profile, _is_epsilon_nash
)


class TestMclennanTourky():
    def setup_method(self):
        def anti_coordination(N, v):
            payoff_array = np.empty((2,)*N)
            payoff_array[0, :] = 1
            payoff_array[1, :] = 0
            payoff_array[1].flat[0] = v
            g = NormalFormGame((Player(payoff_array),)*N)
            return g

        def p_star(N, v):
            # Unique symmetric NE mixed action: [p_star, 1-p_star]
            return 1 / (v**(1/(N-1)))

        def epsilon_nash_interval(N, v, epsilon):
            # Necessary, but not sufficient, condition: lb < p < ub
            lb = p_star(N, v) - epsilon / ((N-1)*(v**(1/(N-1))-1))
            ub = p_star(N, v) + epsilon / (N-1)
            return lb, ub

        self.game_dicts = []
        v = 2
        epsilon = 1e-5

        Ns = [2, 3, 4]
        for N in Ns:
            g = anti_coordination(N, v)
            lb, ub = epsilon_nash_interval(N, v, epsilon)
            d = {'g': g,
                 'epsilon': epsilon,
                 'lb': lb,
                 'ub': ub}
            self.game_dicts.append(d)

    def test_convergence_default(self):
        for d in self.game_dicts:
            NE, res = mclennan_tourky(d['g'], full_output=True)
            assert_(res.converged)

    def test_pure_nash(self):
        for d in self.game_dicts:
            init = (1,) + (0,)*(d['g'].N-1)
            NE, res = mclennan_tourky(d['g'], init=init, full_output=True)
            assert_(res.num_iter == 1)


class TestMclennanTourkyInvalidInputs():
    def setup_method(self):
            self.bimatrix = [[(3, 3), (3, 2)],
                             [(2, 2), (5, 6)],
                             [(0, 3), (6, 1)]]
            self.g = NormalFormGame(self.bimatrix)

    def test_mclennan_tourky_invalid_g(self):
        assert_raises(TypeError, mclennan_tourky, self.bimatrix)

    def test_mclennan_tourky_invalid_init_type(self):
        assert_raises(TypeError, mclennan_tourky, self.g, 1)

    def test_mclennan_tourky_invalid_init_length(self):
        assert_raises(ValueError, mclennan_tourky, self.g, [1])


class TestEpsilonNash():
    def setup_method(self):
        def anti_coordination(N, v):
            payoff_array = np.empty((2,)*N)
            payoff_array[0, :] = 1
            payoff_array[1, :] = 0
            payoff_array[1].flat[0] = v
            g = NormalFormGame((Player(payoff_array),)*N)
            return g

        def p_star(N, v):
            # Unique symmetric NE mixed action: [p_star, 1-p_star]
            return 1 / (v**(1/(N-1)))

        def epsilon_nash_interval(N, v, epsilon):
            # Necessary, but not sufficient, condition: lb < p < ub
            lb = p_star(N, v) - epsilon / ((N-1)*(v**(1/(N-1))-1))
            ub = p_star(N, v) + epsilon / (N-1)
            return lb, ub

        self.game_dicts = []
        v = 2
        epsilon = 1e-5

        Ns = [2, 3, 4]
        for N in Ns:
            g = anti_coordination(N, v)
            lb, ub = epsilon_nash_interval(N, v, epsilon)
            d = {'g': g,
                 'epsilon': epsilon,
                 'lb': lb,
                 'ub': ub}
            self.game_dicts.append(d)

        self.bimatrix = [[(3, 3), (3, 2)],
                         [(2, 2), (5, 6)],
                         [(0, 3), (6, 1)]]
        self.g = NormalFormGame(self.bimatrix)

    def test_epsilon_nash_with_full_output(self):
        for d in self.game_dicts:
            NE, res = \
                mclennan_tourky(d['g'], epsilon=d['epsilon'], full_output=True)
            for i in range(d['g'].N):
                assert_(d['lb'] < NE[i][0] < d['ub'])

    def test_epsilon_nash_without_full_output(self):
        for d in self.game_dicts:
            NE = mclennan_tourky(d['g'], epsilon=d['epsilon'],
                                 full_output=False)
            for i in range(d['g'].N):
                assert_(d['lb'] < NE[i][0] < d['ub'])

    def test_is_epsilon_nash_no_indptr(self):
        assert_(_is_epsilon_nash([1., 0., 0., 1., 0.], self.g, 1e-5))


def test_flatten_action_profile():
    unflattened_actions = [[1/3, 1/3, 1/3], [1/2, 1/2]]
    flattened_actions = [1/3, 1/3, 1/3, 1/2, 1/2]
    test_obj = _flatten_action_profile(unflattened_actions, [0, 3, 5])
    assert_array_equal(test_obj, flattened_actions)


def test_best_response_selection_no_indptr():
    bimatrix = [[(3, 3), (3, 2)],
                [(2, 2), (5, 6)],
                [(0, 3), (6, 1)]]
    g = NormalFormGame(bimatrix)

    test_obj = _best_response_selection([1/3, 1/3, 1/3, 1/2, 1/2], g)
    expected_output = np.array([0., 1., 0., 0., 1.])

    assert_array_equal(test_obj, expected_output)
