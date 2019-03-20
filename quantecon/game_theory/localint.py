import numpy as np
import numbers
from scipy import sparse
from ..util import check_random_state
from .random import random_pure_actions
from .normal_form_game import Player


class LocalInteraction:
    """
    Class representing the local interaction model.

    Parameters
    ----------
    payoff_matrix : array_like(float, ndim=2)
        The payoff matrix of the symmetric two-player game played in
        each interaction.

    adj_matrix : array_like(float, ndim=2)
        The adjacency matrix of the network. Non constant weights and
        asymmetry in interactions are allowed, where adj_matrix[i, j] is
        the weight of player j's action on player i.

    Attributes
    ----------
    players : list(Player)
        The list consisting of all players with the given payoff matrix.

    adj_matrix : scipy.sparse.csr.csr_matrix(float, ndim=2)
        See Parameters.

    N : scalar(int)
        The Number of players.

    num_actions : scalar(int)
        The number of actions available to each player.
    """
    def __init__(self, payoff_matrix, adj_matrix):
        self.adj_matrix = sparse.csr_matrix(adj_matrix)
        M, N = self.adj_matrix.shape
        if N != M:
            raise ValueError('adjacency matrix must be square')
        self.N = N

        A = np.asarray(payoff_matrix)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError('payoff matrix must be square')
        self.num_actions = A.shape[0]

        self.players = [Player(A) for i in range(self.N)]
        self.tie_breaking = 'smallest'

    def _play(self, actions, player_ind=None):
        if player_ind is None:
            player_ind = list(range(self.N))
        elif isinstance(player_ind, numbers.Integral):
            player_ind = [player_ind]

        actions_matrix = sparse.csr_matrix(
            (np.ones(self.N, dtype=int), actions, np.arange(self.N+1)),
            shape=(self.N, self.num_actions))

        opponent_act_dict = self.adj_matrix[player_ind].dot(
            actions_matrix).toarray()

        for k, i in enumerate(player_ind):
            actions[i] = self.players[i].best_response(
                opponent_act_dict[k, :],
                tie_breaking=self.tie_breaking)

        return actions

    def play(self, init_actions=None, player_ind=None, num_reps=1,
             random_state=None):
        """
        Return a new action profile which is updated by playing the game
        `num_reps` times.

        Parameters
        ----------
        init_actions : tuple(int), optional(default=None)
            The action profile in the first period. If None, selected randomly.

        player_ind : array_like(int), optional(default=None)
            The index of players who take actions. If None, all players take
            actions.

        num_reps : scalar(int), optional(default=1)
            The number of iterations.

        Returns
        -------
        tuple(int)
            The action profile after iteration.
        """
        random_state = check_random_state(random_state)
        if init_actions is None:
            nums_actions = tuple([self.num_actions] * self.N)
            init_actions = random_pure_actions(nums_actions, random_state)

        if player_ind is None:
            player_ind = list(range(self.N))
        elif isinstance(player_ind, numbers.Integral):
            player_ind = [player_ind]

        actions = [action for action in init_actions]
        for t in range(num_reps):
            actions = self._play(actions, player_ind)

        return actions

    def time_series(self, ts_length, revision='simultaneous',
                    init_actions=None, player_ind_seq=None, random_state=None):
        """
        Return the array representing time series of each player's actions.

        Parameters
        ----------
        ts_length : scalar(int)
            The number of iterations.

        revision : {'simultaneous', 'asynchronous'}(default='simultaneous')
            The revision method.

        init_actions : tuple(int), optional(default=None)
            The action profile in the first period. If None, selected randomly.

        player_ind_seq : array_like, optional(default=None)
            The sequence of `player_ind`(see `play` Parameters). If None, all
            elements are array designating all players.

        Returns
        -------
        Array
            The array representing time series of each player's actions.
        """
        random_state = check_random_state(random_state)
        if init_actions is None:
            nums_actions = tuple([self.num_actions] * self.N)
            init_actions = random_pure_actions(nums_actions, random_state)

        if revision == 'simultaneous':
            player_ind_seq = [None] * ts_length
        elif revision == 'asynchronous':
            if player_ind_seq is None:
                player_ind_seq = random_state.randint(self.N, size=ts_length)
        else:
            raise ValueError(
                            "revision must be `simultaneous` or `asynchronous`"
                            )

        actions = [action for action in init_actions]
        out = np.empty((ts_length, self.N), dtype=int)
        for t in range(ts_length):
            for i in range(self.N):
                out[t, i] = actions[i]
            actions = self._play(actions, player_ind_seq[t])

        return out
