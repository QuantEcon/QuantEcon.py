"""
Generate random NormalFormGame instances.

"""

import numpy as np

from .normal_form_game import Player, NormalFormGame
from ..util import check_random_state
from ..random import probvec


def random_game(nums_actions, random_state=None):
    """
    Return a random NormalFormGame instance where the payoffs are drawn
    independently from the uniform distribution on [0, 1).

    Parameters
    ----------
    nums_actions : tuple(int)
        Tuple of the numbers of actions, one for each player.

    random_state : int or np.random.RandomState, optional
        Random seed (integer) or np.random.RandomState instance to set
        the initial state of the random number generator for
        reproducibility. If None, a randomly initialized RandomState is
        used.

    Returns
    -------
    g : NormalFormGame

    """
    N = len(nums_actions)
    if N == 0:
        raise ValueError('nums_actions must be non-empty')

    random_state = check_random_state(random_state)
    players = [
        Player(random_state.random_sample(nums_actions[i:]+nums_actions[:i]))
        for i in range(N)
    ]
    g = NormalFormGame(players)
    return g


def covariance_game(nums_actions, rho, random_state=None):
    """
    Return a random NormalFormGame instance where the payoff profiles
    are drawn independently from the standard multi-normal with the
    covariance of any pair of payoffs equal to `rho`, as studied in
    [1]_.

    Parameters
    ----------
    nums_actions : tuple(int)
        Tuple of the numbers of actions, one for each player.

    rho : scalar(float)
        Covariance of a pair of payoff values. Must be in [-1/(N-1), 1],
        where N is the number of players.

    random_state : int or np.random.RandomState, optional
        Random seed (integer) or np.random.RandomState instance to set
        the initial state of the random number generator for
        reproducibility. If None, a randomly initialized RandomState is
        used.

    Returns
    -------
    g : NormalFormGame

    References
    ----------
    .. [1] Y. Rinott and M. Scarsini, "On the Number of Pure Strategy
       Nash Equilibria in Random Games," Games and Economic Behavior
       (2000), 274-293.

    """
    N = len(nums_actions)
    if N <= 1:
        raise ValueError('length of nums_actions must be at least 2')
    if not (-1 / (N - 1) <= rho <= 1):
        lb = '-1' if N == 2 else '-1/{0}'.format(N-1)
        raise ValueError('rho must be in [{0}, 1]'.format(lb))

    mean = np.zeros(N)
    cov = np.empty((N, N))
    cov.fill(rho)
    cov[range(N), range(N)] = 1

    random_state = check_random_state(random_state)
    payoff_profile_array = \
        random_state.multivariate_normal(mean, cov, nums_actions)
    g = NormalFormGame(payoff_profile_array)
    return g


def random_pure_actions(nums_actions, random_state=None):
    """
    Return a tuple of random pure actions (integers).

    Parameters
    ----------
    nums_actions : tuple(int)
        Tuple of the numbers of actions, one for each player.

    random_state : int or np.random.RandomState, optional
        Random seed (integer) or np.random.RandomState instance to set
        the initial state of the random number generator for
        reproducibility. If None, a randomly initialized RandomState is
        used.

    Returns
    -------
    action_profile : Tuple(int)
        Tuple of actions, one for each player.

    """
    random_state = check_random_state(random_state)
    action_profile = tuple(
        [random_state.randint(num_actions) for num_actions in nums_actions]
    )
    return action_profile


def random_mixed_actions(nums_actions, random_state=None):
    """
    Return a tuple of random mixed actions (vectors of floats).

    Parameters
    ----------
    nums_actions : tuple(int)
        Tuple of the numbers of actions, one for each player.

    random_state : int or np.random.RandomState, optional
        Random seed (integer) or np.random.RandomState instance to set
        the initial state of the random number generator for
        reproducibility. If None, a randomly initialized RandomState is
        used.

    Returns
    -------
    action_profile : tuple(ndarray(float, ndim=1))
        Tuple of mixed_actions, one for each player.

    """
    random_state = check_random_state(random_state)
    action_profile = tuple(
        [probvec(1, num_actions, random_state).ravel()
         for num_actions in nums_actions]
    )
    return action_profile
