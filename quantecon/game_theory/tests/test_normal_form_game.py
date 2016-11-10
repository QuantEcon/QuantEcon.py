"""
Author: Daisuke Oyama

Tests for normal_form_game.py

"""
from __future__ import division

import numpy as np
from numpy.testing import assert_array_equal
from nose.tools import eq_, ok_, raises

from quantecon.game_theory import (
    Player, NormalFormGame, pure2mixed, best_response_2p
)


# Player #

class TestPlayer_1opponent:
    """Test the methods of Player with one opponent player"""

    def setUp(self):
        """Setup a Player instance"""
        coordination_game_matrix = [[4, 0], [3, 2]]
        self.player = Player(coordination_game_matrix)

    def test_best_response_against_pure(self):
        eq_(self.player.best_response(1), 1)

    def test_best_response_against_mixed(self):
        eq_(self.player.best_response([1/2, 1/2]), 1)

    def test_best_response_list_when_tie(self):
        """best_response with tie_breaking=False"""
        assert_array_equal(
            sorted(self.player.best_response([2/3, 1/3], tie_breaking=False)),
            sorted([0, 1])
        )

    def test_best_response_with_random_tie_breaking(self):
        """best_response with tie_breaking='random'"""
        ok_(self.player.best_response([2/3, 1/3], tie_breaking='random')
            in [0, 1])

        seed = 1234
        br0 = self.player.best_response([2/3, 1/3], tie_breaking='random',
                                        random_state=seed)
        br1 = self.player.best_response([2/3, 1/3], tie_breaking='random',
                                        random_state=seed)
        eq_(br0, br1)

    def test_best_response_with_smallest_tie_breaking(self):
        """best_response with tie_breaking='smallest' (default)"""
        eq_(self.player.best_response([2/3, 1/3]), 0)

    def test_best_response_with_payoff_perturbation(self):
        """best_response with payoff_perturbation"""
        eq_(self.player.best_response([2/3, 1/3],
                                      payoff_perturbation=[0, 0.1]),
            1)
        eq_(self.player.best_response([2, 1],  # int
                                      payoff_perturbation=[0, 0.1]),
            1)

    def test_is_best_response_against_pure(self):
        ok_(self.player.is_best_response(0, 0))

    def test_is_best_response_against_mixed(self):
        ok_(self.player.is_best_response([1/2, 1/2], [2/3, 1/3]))


class TestPlayer_2opponents:
    """Test the methods of Player with two opponent players"""

    def setUp(self):
        """Setup a Player instance"""
        payoffs_2opponents = [[[3, 6],
                               [4, 2]],
                              [[1, 0],
                               [5, 7]]]
        self.player = Player(payoffs_2opponents)

    def test_payoff_vector_against_pure(self):
        assert_array_equal(self.player.payoff_vector((0, 1)), [6, 0])

    def test_is_best_response_against_pure(self):
        ok_(not self.player.is_best_response(0, (1, 0)))

    def test_best_response_against_pure(self):
        eq_(self.player.best_response((1, 1)), 1)

    def test_best_response_list_when_tie(self):
        """
        best_response against a mixed action profile with
        tie_breaking=False
        """
        assert_array_equal(
            sorted(self.player.best_response(([3/7, 4/7], [1/2, 1/2]),
                                             tie_breaking=False)),
            sorted([0, 1])
        )


def test_random_choice():
    n, m = 5, 4
    payoff_matrix = np.zeros((n, m))
    player = Player(payoff_matrix)

    eq_(player.random_choice([0]), 0)

    actions = list(range(player.num_actions))
    ok_(player.random_choice() in actions)


# NormalFormGame #

class TestNormalFormGame_Sym2p:
    """Test the methods of NormalFormGame with symmetric two players"""

    def setUp(self):
        """Setup a NormalFormGame instance"""
        coordination_game_matrix = [[4, 0], [3, 2]]
        self.g = NormalFormGame(coordination_game_matrix)

    def test_getitem(self):
        assert_array_equal(self.g[0, 1], [0, 3])

    def test_is_nash_pure(self):
        ok_(self.g.is_nash((0, 0)))

    def test_is_nash_mixed(self):
        ok_(self.g.is_nash(([2/3, 1/3], [2/3, 1/3])))


class TestNormalFormGame_Asym2p:
    """Test the methods of NormalFormGame with asymmetric two players"""

    def setUp(self):
        """Setup a NormalFormGame instance"""
        matching_pennies_bimatrix = [[(1, -1), (-1, 1)],
                                     [(-1, 1), (1, -1)]]
        self.g = NormalFormGame(matching_pennies_bimatrix)

    def test_getitem(self):
        assert_array_equal(self.g[1, 0], [-1, 1])

    def test_is_nash_against_pure(self):
        ok_(not self.g.is_nash((0, 0)))

    def test_is_nash_against_mixed(self):
        ok_(self.g.is_nash(([1/2, 1/2], [1/2, 1/2])))


class TestNormalFormGame_3p:
    """Test the methods of NormalFormGame with three players"""

    def setUp(self):
        """Setup a NormalFormGame instance"""
        payoffs_2opponents = [[[3, 6],
                               [4, 2]],
                              [[1, 0],
                               [5, 7]]]
        player = Player(payoffs_2opponents)
        self.g = NormalFormGame([player for i in range(3)])

    def test_getitem(self):
        assert_array_equal(self.g[0, 0, 1], [6, 4, 1])

    def test_is_nash_pure(self):
        ok_(self.g.is_nash((0, 0, 0)))
        ok_(not self.g.is_nash((0, 0, 1)))

    def test_is_nash_mixed(self):
        p = (1 + np.sqrt(65)) / 16
        ok_(self.g.is_nash(([1 - p, p], [1 - p, p], [1 - p, p])))


def test_normalformgame_input_action_sizes():
    g = NormalFormGame((2, 3, 4))

    eq_(g.N, 3)  # Number of players

    assert_array_equal(
        g.players[0].payoff_array,
        np.zeros((2, 3, 4))
    )
    assert_array_equal(
        g.players[1].payoff_array,
        np.zeros((3, 4, 2))
    )
    assert_array_equal(
        g.players[2].payoff_array,
        np.zeros((4, 2, 3))
    )


def test_normalformgame_setitem():
    g = NormalFormGame((2, 2))
    g[0, 0] = (0, 10)
    g[0, 1] = (0, 10)
    g[1, 0] = (3, 5)
    g[1, 1] = (-2, 0)

    assert_array_equal(
        g.players[0].payoff_array,
        [[0, 0], [3, -2]]
    )
    assert_array_equal(
        g.players[1].payoff_array,
        [[10, 5], [10, 0]]
    )


def test_normalformgame_constant_payoffs():
    g = NormalFormGame((2, 2))

    ok_(g.is_nash((0, 0)))
    ok_(g.is_nash((0, 1)))
    ok_(g.is_nash((1, 0)))
    ok_(g.is_nash((1, 1)))


def test_normalformgame_payoff_profile_array():
    nums_actions = (2, 3, 4)
    for N in range(1, len(nums_actions)+1):
        payoff_arrays = [
            np.arange(np.prod(nums_actions[0:N])).reshape(nums_actions[i:N] +
                                                          nums_actions[0:i])
            for i in range(N)
        ]
        players = [Player(payoff_array) for payoff_array in payoff_arrays]
        g = NormalFormGame(players)
        g_new = NormalFormGame(g.payoff_profile_array)
        for player_new, payoff_array in zip(g_new.players, payoff_arrays):
            assert_array_equal(player_new.payoff_array, payoff_array)


def test_normalformgame_payoff_profile_array_c_contiguous():
    nums_actions = (2, 3, 4)
    shape = nums_actions + (len(nums_actions),)
    payoff_profile_array = \
        np.arange(np.prod(shape)).reshape(shape)
    g = NormalFormGame(payoff_profile_array)
    for player in g.players:
        ok_(player.payoff_array.flags['C_CONTIGUOUS'])


# Trivial cases with one player #

class TestPlayer_0opponents:
    """Test for trivial Player with no opponent player"""

    def setUp(self):
        """Setup a Player instance"""
        payoffs = [0, 1]
        self.player = Player(payoffs)

    def test_payoff_vector(self):
        """Trivial player: payoff_vector"""
        assert_array_equal(self.player.payoff_vector(None), [0, 1])

    def test_is_best_response(self):
        """Trivial player: is_best_response"""
        ok_(self.player.is_best_response(1, None))

    def test_best_response(self):
        """Trivial player: best_response"""
        eq_(self.player.best_response(None), 1)


class TestNormalFormGame_1p:
    """Test for trivial NormalFormGame with a single player"""

    def setUp(self):
        """Setup a NormalFormGame instance"""
        data = [[0], [1], [1]]
        self.g = NormalFormGame(data)

    def test_construction(self):
        """Trivial game: construction"""
        ok_(self.g.N == 1)
        assert_array_equal(self.g.players[0].payoff_array, [0, 1, 1])

    def test_getitem(self):
        """Trivial game: __getitem__"""
        eq_(self.g[0], 0)

    def test_is_nash_pure(self):
        """Trivial game: is_nash with pure action"""
        ok_(self.g.is_nash((1,)))
        ok_(not self.g.is_nash((0,)))

    def test_is_nash_mixed(self):
        """Trivial game: is_nash with mixed action"""
        ok_(self.g.is_nash(([0, 1/2, 1/2],)))


def test_normalformgame_input_action_sizes_1p():
    g = NormalFormGame(2)

    eq_(g.N, 1)  # Number of players

    assert_array_equal(
        g.players[0].payoff_array,
        np.zeros(2)
    )


def test_normalformgame_setitem_1p():
    g = NormalFormGame(2)

    eq_(g.N, 1)  # Number of players

    g[0] = 10  # Set payoff 10 for action 0
    eq_(g.players[0].payoff_array[0], 10)


# Test __repre__ #

def test_player_repr():
    nums_actions = (2, 3, 4)
    payoff_arrays = [
        np.arange(np.prod(nums_actions[0:i])).reshape(nums_actions[0:i])
        for i in range(1, len(nums_actions)+1)
    ]
    players = [Player(payoff_array) for payoff_array in payoff_arrays]

    for player in players:
        player_new = eval(repr(player))
        assert_array_equal(player_new.payoff_array, player.payoff_array)


# Invalid inputs #

@raises(ValueError)
def test_normalformgame_invalid_input_players_shape_inconsistent():
    p0 = Player(np.zeros((2, 3)))
    p1 = Player(np.zeros((2, 3)))
    g = NormalFormGame([p0, p1])


@raises(ValueError)
def test_normalformgame_invalid_input_players_num_inconsistent():
    p0 = Player(np.zeros((2, 2, 2)))
    p1 = Player(np.zeros((2, 2, 2)))
    g = NormalFormGame([p0, p1])


@raises(ValueError)
def test_normalformgame_invalid_input_players_dtype_inconsistent():
    p0 = Player(np.zeros((2, 2), dtype=int))
    p1 = Player(np.zeros((2, 2), dtype=float))
    g = NormalFormGame([p0, p1])


@raises(ValueError)
def test_normalformgame_invalid_input_nosquare_matrix():
    g = NormalFormGame(np.zeros((2, 3)))


@raises(ValueError)
def test_normalformgame_invalid_input_payoff_profiles():
    g = NormalFormGame(np.zeros((2, 2, 1)))


# Utility functions #

def test_pure2mixed():
    num_actions = 3
    pure_action = 0
    mixed_action = [1., 0., 0.]

    assert_array_equal(pure2mixed(num_actions, pure_action), mixed_action)


# Numba jitted functions #

def test_best_response_2p():
    test_case0 = {
        'payoff_array': np.array([[4, 0], [3, 2], [0, 3]]),
        'mixed_actions':
        [np.array([1, 0]), np.array([0.5, 0.5]), np.array([0, 1])],
        'brs_expected': [0, 1, 2]
    }
    test_case1 = {
        'payoff_array': np.zeros((2, 3)),
        'mixed_actions': [np.array([1, 0, 0]), np.array([1/3, 1/3, 1/3])],
        'brs_expected': [0, 0]
    }

    for test_case in [test_case0, test_case1]:
        for mixed_action, br_expected in zip(test_case['mixed_actions'],
                                             test_case['brs_expected']):
            br_computed = \
                best_response_2p(test_case['payoff_array'], mixed_action)
            eq_(br_computed, br_expected)


if __name__ == '__main__':
    import sys
    import nose

    argv = sys.argv[:]
    argv.append('--verbose')
    argv.append('--nocapture')
    nose.main(argv=argv, defaultTest=__file__)
