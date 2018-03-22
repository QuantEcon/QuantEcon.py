"""
Tests for game_theory/random.py

"""
import numpy as np
from numpy.testing import assert_allclose, assert_raises
from nose.tools import eq_

from quantecon.game_theory import random_game, covariance_game


def test_random_game():
    nums_actions = (2, 3, 4)
    g = random_game(nums_actions)
    eq_(g.nums_actions, nums_actions)


def test_covariance_game():
    nums_actions = (2, 3, 4)
    N = len(nums_actions)

    rho = 0.5
    g = covariance_game(nums_actions, rho=rho)
    eq_(g.nums_actions, nums_actions)

    rho = 1
    g = covariance_game(nums_actions, rho=rho)
    for a in np.ndindex(*nums_actions):
        for i in range(N-1):
            payoff_profile = g.payoff_profile_array[a]
            assert_allclose(payoff_profile[i], payoff_profile[-1], atol=1e-8)

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


if __name__ == '__main__':
    import sys
    import nose

    argv = sys.argv[:]
    argv.append('--verbose')
    argv.append('--nocapture')
    nose.main(argv=argv, defaultTest=__file__)
