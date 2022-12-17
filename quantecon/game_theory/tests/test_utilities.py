"""
Tests for game_theory/utilities.py

"""
import numpy as np
from numpy.testing import assert_array_equal
from quantecon.game_theory.utilities import (
    _copy_action_to, _copy_action_profile_to
)


def test_copy_action_to():
    pure_action = 1
    n = 3
    mixed_action = np.zeros(n)
    mixed_action[pure_action] = 1
    dst = np.empty(n)
    _copy_action_to(dst, pure_action)
    assert_array_equal(dst, mixed_action)

    mixed_action = np.array([0.2, 0.3, 0.5])
    n = len(mixed_action)
    dst = np.empty(n)
    _copy_action_to(dst, mixed_action)
    assert_array_equal(dst, mixed_action)


def test_copy_action_profile_to():
    pure_action = 1
    n0 = 3
    mixed_action0 = np.zeros(n0)
    mixed_action0[pure_action] = 1

    action_profile = (pure_action, np.array([0.2, 0.3, 0.5]), [0.4, 0.6])
    mixed_actions = (mixed_action0,) + action_profile[1:]
    nums_actions = tuple(len(x) for x in mixed_actions)
    dst = tuple(np.empty(n) for n in nums_actions)

    _copy_action_profile_to(dst, mixed_actions)
    for x, y in zip(dst, mixed_actions):
        assert_array_equal(x, y)


if __name__ == '__main__':
    import sys
    import nose

    argv = sys.argv[:]
    argv.append('--verbose')
    argv.append('--nocapture')
    nose.main(argv=argv, defaultTest=__file__)
