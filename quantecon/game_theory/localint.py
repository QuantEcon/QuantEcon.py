import numpy as np
import numbers
from scipy import sparse
from ..util import check_random_state, rng_integers
from .random import random_pure_actions
from .normal_form_game import Player


class LocalInteraction:
    """
    Class representing the local interaction model.

    Parameters
    ----------
    payoff_matrix : array_like(float, ndim=2)
        The payoff matrix of the symmetric two-player game played in each
        interaction.

    adj_matrix : array_like(float, ndim=2)
        The adjacency matrix of the network. Non constant weights and asymmetry
        in interactions are allowed.

    Attributes
    ----------
    players : list(Player)
        The list consisting of all players with the given payoff matrix.

    adj_matrix : scipy.sparse.csr.csr_matrix(float, ndim=2)
        See Parameters.

    N : scalar(int)
        The number of players.

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

    def _play(self, actions, player_ind, tie_breaking, tol, random_state):
        actions_matrix = sparse.csr_matrix(
            (np.ones(self.N, dtype=int), actions, np.arange(self.N+1)),
            shape=(self.N, self.num_actions))

        opponent_act_dict = self.adj_matrix[player_ind].dot(
            actions_matrix).toarray()

        for k, i in enumerate(player_ind):
            actions[i] = self.players[i].best_response(
                opponent_act_dict[k, :], tie_breaking=tie_breaking,
                tol=tol, random_state=random_state
            )

        return actions

    def play(self, revision='simultaneous', actions=None,
             player_ind_seq=None, num_reps=1, **options):
        """
        Return a new action profile which is updated by playing the game
        `num_reps` times.

        Parameters
        ----------
        revision : str, optional(default='simultaneous')
            The way to revise the action profile. If `simulataneous`, all
            players' actions will be updated simultaneously. If `asynchronous`,
            only designated players' actions will be updated. str in
            {'simultaneous', 'asynchronous'}.

        actions : tuple(int) or list(int), optional(default=None)
            The action profile in the first period. If None, selected randomly.

        player_ind_seq : array_like(int), optional(default=None)
            The index of players who take actions. If None, all players take
            actions.

        num_reps : scalar(int), optional(default=1)
            The number of iterations.

        **options : Keyword arguments passed to the best response method and
                    other methods.

        Returns
        -------
        tuple(int)
            The action profile after iterations.

        """
        tie_breaking = options.get('tie_breaking', self.tie_breaking)
        tol = options.get('tol', None)
        random_state = check_random_state(options.get('random_state', None))

        if actions is None:
            nums_actions = tuple([self.num_actions] * self.N)
            actions = random_pure_actions(nums_actions, random_state)

        if revision == 'simultaneous':
            player_ind_seq = [None] * num_reps
        elif revision == 'asynchronous':
            if player_ind_seq is None:
                random_state = check_random_state(random_state)
                player_ind_seq = rng_integers(random_state, self.N,
                                              size=num_reps)
            elif isinstance(player_ind_seq, numbers.Integral):
                player_ind_seq = [player_ind_seq]
        else:
            raise ValueError(
                            "revision must be `simultaneous` or `asynchronous`"
                            )

        actions = list(actions)
        for t, player_ind in enumerate(player_ind_seq):
            if player_ind is None:
                player_ind = list(range(self.N))
            elif isinstance(player_ind, numbers.Integral):
                player_ind = [player_ind]
            actions = self._play(actions, player_ind, tie_breaking,
                                 tol, random_state)

        return tuple(actions)

    def time_series(self, ts_length, revision='simultaneous',
                    actions=None, player_ind_seq=None, **options):
        """
        Return an array representing time series of each player's actions.

        Parameters
        ----------
        ts_length : scalar(int)
            The number of iterations.

        revision : {'simultaneous', 'asynchronous'}(default='simultaneous')
            The way to revise the action profile. If `simulataneous`, all
            players' actions will be updated simultaneously. If `asynchronous`,
            only designated players' actions will be updated. str in
            {'simultaneous', 'asynchronous'}.

        actions : tuple(int), optional(default=None)
            The action profile in the first period. If None, selected randomly.

        player_ind_seq : array_like, optional(default=None)
            The sequence of `player_ind`(see `play` Parameters) when the
            revision is 'asynchronous'. If None, selected randomly.

        **options : Keyword arguments passed to the best response method and
                    other methods.

        Returns
        -------
        Array_like(int)
            The array representing time series of each player's actions.

        """
        tie_breaking = options.get('tie_breaking', self.tie_breaking)
        tol = options.get('tol', None)
        random_state = check_random_state(options.get('random_state', None))

        if actions is None:
            nums_actions = tuple([self.num_actions] * self.N)
            actions = random_pure_actions(nums_actions, random_state)

        if revision == 'simultaneous':
            player_ind_seq = [None] * ts_length
        elif revision == 'asynchronous':
            if player_ind_seq is None:
                random_state = check_random_state(random_state)
                player_ind_seq = rng_integers(random_state, self.N,
                                              size=ts_length)
        else:
            raise ValueError(
                            "revision must be `simultaneous` or `asynchronous`"
                            )

        out = np.empty((ts_length, self.N), dtype=int)
        for i in range(self.N):
            out[0, i] = actions[i]
        for t in range(ts_length-1):
            random_state = check_random_state(random_state)
            actions = self.play(revision=revision,
                                actions=actions,
                                player_ind_seq=player_ind_seq[t],
                                num_reps=1,
                                tie_breaking=tie_breaking,
                                tol=tol, random_state=random_state)
            for i in range(self.N):
                out[t+1, i] = actions[i]

        return out
