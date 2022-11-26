import numpy as np
import numbers
from .normal_form_game import NormalFormGame
from ..util import check_random_state, rng_integers
from .random import random_pure_actions


class LogitDynamics:
    """
    Class representing the logit-response dynamics model.

    Parameters
    ----------
    data : NormalFormGame or array_like
        The game played in the logit-response dynamics model.

    beta : scalar(float)
        The level of noise in player's decision.

    Attributes
    ----------
    N : scalar(int)
        The number of players in the game.

    players : list(Player)
        The list consisting of all players with the given payoff matrix.

    nums_actions : tuple(int)
        Tuple of the number of actions, one for each player.

    beta : scalar(float)
        See parameters.

    player.logit_choice_cdfs : array_like(float)
        The choice probability of each actions given opponents' actions.

    """
    def __init__(self, data, beta=1.0):
        if isinstance(data, NormalFormGame):
            self.g = data
        else:  # data must be array_like
            self.g = NormalFormGame(data)

        self.N = self.g.N
        self.players = self.g.players
        self.nums_actions = self.g.nums_actions

        self.beta = beta

        for player in self.players:
            payoff_array_rotated = \
                player.payoff_array.transpose(list(range(1, self.N)) + [0])
            payoff_array_rotated -= \
                payoff_array_rotated.max(axis=-1)[..., np.newaxis]
            player.logit_choice_cdfs = \
                np.exp(payoff_array_rotated*self.beta).cumsum(axis=-1)

    def logit_choice_cdfs(self):
        """
        Return the tuple of choice probabilities.

        """
        return tuple(player.logit_choice_cdfs for player in self.players)

    def _play(self, player_ind, actions, random_state):
        i = player_ind

        # Tuple of the actions of opponent players i+1, ..., N, 0, ..., i-1
        opponent_actions = \
            tuple(actions[i+1:]) + tuple(actions[:i])

        cdf = self.players[i].logit_choice_cdfs[opponent_actions]
        random_value = random_state.random()
        next_action = cdf.searchsorted(random_value*cdf[-1], side='right')

        return next_action

    def play(self, init_actions=None, player_ind_seq=None, num_reps=1,
             random_state=None):
        """
        Return a new action profile which is updated `num_reps` times.

        Parameters
        ----------
        init_actions : tuple(int), optional(default=None)
            The action profile in the initial period. If None, selected
            randomly.

        player_ind_seq : list(int), optional(default=None)
            The sequence of player indices. If None, selected randomly.

        num_reps : scalar(int), optional(default=1)
            The number of iterations.

        random_state : int or np.random.RandomState/Generator, optional
            Random number generator used.

        Returns
        -------
        tuple(int)
            The action profile after iterations.

        """
        random_state = check_random_state(random_state)

        if init_actions is None:
            init_actions = random_pure_actions(self.nums_actions, random_state)
        init_actions = list(init_actions)

        if player_ind_seq is None:
            random_state = check_random_state(random_state)
            player_ind_seq = rng_integers(random_state, self.N, size=num_reps)

        if isinstance(player_ind_seq, numbers.Integral):
            player_ind_seq = [player_ind_seq]

        for t, player_ind in enumerate(player_ind_seq):
            random_state = check_random_state(random_state)
            init_actions[player_ind] = self._play(player_ind, init_actions,
                                                  random_state)

        return tuple(init_actions)

    def time_series(self, ts_length, init_actions=None, random_state=None):
        """
        Return the array representing time series of action profiles.

        Parameters
        ----------
        ts_length : scalar(int)
            The number of iterations.

        init_actions : tuple(int), optional(default=None)
            The action profile in the initial period. If None, selected
            randomly.

        random_state : int or np.random.RandomState/Generator, optional
            Random number generator used.

        Returns
        -------
        ndarray(int)
            The array representing the time series of action profiles.

        """
        if init_actions is None:
            random_state = check_random_state(random_state)
            init_actions = random_pure_actions(self.nums_actions, random_state)
        actions = list(init_actions)

        out = np.empty((ts_length, self.N), dtype=int)
        random_state = check_random_state(random_state)
        player_ind_seq = rng_integers(random_state, self.N, size=ts_length)

        for t, player_ind in enumerate(player_ind_seq):
            out[t, :] = actions[:]
            random_state = check_random_state(random_state)
            actions[player_ind] = self._play(player_ind, actions, random_state)

        return out
