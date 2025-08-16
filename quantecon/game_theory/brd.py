import numpy as np
from .normal_form_game import Player
from ..util import check_random_state, rng_integers
from .random import random_pure_actions


class BRD:
    """
    Class representing the best response dynamics model.

    Parameters
    ----------
    payoff_matrix : array_like(float, ndim=2)
        The payoff matrix of the symmetric two-player game.

    N : scalar(int)
        The number of players.

    Attributes
    ----------
    N : scalar(int)
        The number of players.

    num_actions : scalar(int)
        The number of actions.

    player : Player
        Player instance in the model.

    """
    def __init__(self, payoff_matrix, N):
        A = np.asarray(payoff_matrix)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError('payoff matrix must be square')
        self.num_actions = A.shape[0]  # Number of actions
        self.N = N  # Number of players

        self.player = Player(A)  # "Representative player"
        self.tie_breaking = 'smallest'

    def _set_action_dist(self, actions):
        if self.N != len(actions):
            raise ValueError("length of `actions` must equal to the number of \
                              players")
        action_dist = np.zeros(self.num_actions)
        for i in range(self.N):
            action_dist[actions[i]] += 1
        return action_dist

    def play(self, action, action_dist, **options):
        """
        Return a new action distribution.

        Parameters
        ----------
        action : scalar(int)
            Pure action of player who takes action.

        action_dist : ndarray(int)
            The initial action distribution of players.

        **options : Keyword arguments passed to the best response method.

        Returns
        -------
        ndarray(int)
            New action distribution.

        """
        tie_breaking = options.get('tie_breaking', self.tie_breaking)
        tol = options.get('tol', None)
        random_state = check_random_state(options.get('random_state', None))

        action_dist[action] -= 1
        next_action = self.player.best_response(action_dist,
                                                tol=tol,
                                                tie_breaking=tie_breaking,
                                                random_state=random_state)
        action_dist[next_action] += 1
        return action_dist

    def time_series(self, ts_length, init_action_dist=None, **options):
        """
        Return a time series of action distributions. The order of player who
        takes a action is randomly choosed.

        Parameters
        ----------
        ts_length : scalar(int)
            The length of time series.

        init_action_dist : array_like(int), optional(default=None)
            The initial action distribution. If None, determined randomly.

        **options : Keyword arguments passed to the best response method and
                    other methods.

        Returns
        -------
        ndarray(int)
            The array representing time series of action distributions.

        """
        tie_breaking = options.get('tie_breaking', self.tie_breaking)
        tol = options.get('tol', None)
        random_state = check_random_state(options.get('random_state', None))

        if init_action_dist is None:
            nums_actions = tuple([self.num_actions] * self.N)
            init_actions = random_pure_actions(nums_actions, random_state)
            init_action_dist = self._set_action_dist(init_actions)

        out = np.empty((ts_length, self.num_actions), dtype=int)
        random_state = check_random_state(random_state)
        player_ind_seq = rng_integers(random_state, self.N, size=ts_length)
        action_dist = np.asarray(init_action_dist)
        for t in range(ts_length):
            out[t, :] = action_dist[:]
            action = np.searchsorted(action_dist.cumsum(), player_ind_seq[t],
                                     side='right')
            random_state = check_random_state(random_state)
            action_dist = self.play(action, action_dist,
                                    tol=tol,
                                    tie_breaking=tie_breaking,
                                    random_state=random_state)
        return out


class KMR(BRD):
    """
    Class representing the Kandori-Mailath-Rob model.

    Parameters
    ----------
    payoff_matrix : array_like(float, ndim=2)
        The payoff matrix of the symmetric two-player game.

    N : scalar(int)
        The number of players.

    epsilon : scalar(float), default=0.1
        The probability of strategy flips.

    Attributes
    ----------
    N : scalar(int)
        See parameters.

    num_actions : scalar(int)
        The number of actions.

    player : Player
        Player instance in the model.

    epsilon : scalar(float)
        See parameters.

    """
    def __init__(self, payoff_matrix, N, epsilon=0.1):
        BRD.__init__(self, payoff_matrix, N)

        # Mutation probability
        self.epsilon = epsilon

    def play(self, action, action_dist, **options):
        """
        See `play` in `BRD`.

        """
        tie_breaking = options.get('tie_breaking', self.tie_breaking)
        tol = options.get('tol', None)
        random_state = check_random_state(options.get('random_state', None))

        if random_state.random() < self.epsilon:  # Mutation
            action_dist[action] -= 1
            random_state = check_random_state(random_state)
            next_action = self.player.random_choice(random_state=random_state)
            action_dist[next_action] += 1
        else:  # Best response
            random_state = check_random_state(random_state)
            action_dist = BRD.play(self, action, action_dist,
                                   tol=tol,
                                   tie_breaking=tie_breaking,
                                   random_state=random_state)
        return action_dist


class SamplingBRD(BRD):
    """
    Class representing the sampling BRD model.

    Parameters
    ----------
    payoff_matrix : array_like(float, ndim=2)
        The payoff matrix of the symmetric two-player game.

    N : scalar(int)
        The number of players.

    k : scalar(int), default=2
        Sample size.

    Attributes
    ----------
    N : scalar(int)
        See parameters.

    num_actions : scalar(int)
        The number of actions.

    player : Player
        Player instance in the model.

    k : scalar(int), default=2
        See parameters.

    """
    def __init__(self, payoff_matrix, N, k=2):
        BRD.__init__(self, payoff_matrix, N)

        # Sample size
        self.k = k

    def play(self, action, action_dist, **options):
        """
        See `play` in `BRD`.

        """
        tie_breaking = options.get('tie_breaking', self.tie_breaking)
        tol = options.get('tol', None)
        random_state = check_random_state(options.get('random_state', None))

        action_dist[action] -= 1
        actions = random_state.choice(self.num_actions, size=self.k,
                                      replace=True, p=action_dist/(self.N-1))
        sample_action_dist = np.bincount(actions, minlength=self.num_actions)
        random_state = check_random_state(random_state)
        next_action = self.player.best_response(sample_action_dist,
                                                tol=tol,
                                                tie_breaking=tie_breaking,
                                                random_state=random_state)
        action_dist[next_action] += 1
        return action_dist
