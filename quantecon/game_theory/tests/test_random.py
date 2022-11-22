"""
Tests for game_theory/random.py

"""
import numpy as np
from numpy.testing import assert_allclose, assert_raises, assert_
from quantecon.game_theory import (
    random_game, covariance_game, random_pure_actions, random_mixed_actions
)


def test_random_game():
    nums_actions = (2, 3, 4)
    g = random_game(nums_actions)
    assert_(g.nums_actions == nums_actions)


def test_covariance_game():
    nums_actions = (2, 3, 4)
    N = len(nums_actions)

    rho = 0.5
    g = covariance_game(nums_actions, rho=rho)
    assert_(g.nums_actions == nums_actions)

    rho = 1
    g = covariance_game(nums_actions, rho=rho)
    for a in np.ndindex(*nums_actions):
        payoff_profile = g.payoff_profile_array[a]
        for i in range(N-1):
            assert_allclose(payoff_profile[i], payoff_profile[-1], atol=1e-7)

    rho = -1 / (N - 1)
    g = covariance_game(nums_actions, rho=rho)
    for a in np.ndindex(*nums_actions):
        assert_allclose(g.payoff_profile_array.sum(axis=-1),
                        np.zeros(nums_actions),
                        atol=1e-10)


def test_random_game_value_error():
    nums_actions = ()  # empty
    assert_raises(ValueError, random_game, nums_actions)


def test_covariance_game_value_error():
    nums_actions = ()  # empty
    assert_raises(ValueError, covariance_game, nums_actions, rho=0)

    nums_actions = (2,)  # length one
    assert_raises(ValueError, covariance_game, nums_actions, rho=0)

    nums_actions = (2, 3, 4)

    rho = 1.1  # > 1
    assert_raises(ValueError, covariance_game, nums_actions, rho)

    rho = -1  # < -1/(N-1)
    assert_raises(ValueError, covariance_game, nums_actions, rho)


def test_random_pure_actions():
    nums_actions = (2, 3, 4)
    N = len(nums_actions)
    seed = 1234
    action_profiles = [
        random_pure_actions(nums_actions, seed) for i in range(2)
    ]
    for i in range(N):
        assert_(action_profiles[0][i] < nums_actions[i])
    assert_(action_profiles[0] == action_profiles[1])


def test_random_mixed_actions():
    nums_actions = (2, 3, 4)
    seed = 1234
    action_profile = random_mixed_actions(nums_actions, seed)
    assert_(tuple([len(action) for action in action_profile]) == nums_actions)
