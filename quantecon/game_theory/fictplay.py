import numpy as np
from .normal_form_game import NormalFormGame, pure2mixed
from ..util import check_random_state
from .random import random_pure_actions


class FictitiousPlay:
    """
    Class representing the fictitious play model.

    Parameters
    ----------
    data : NormalFormGame or array-like
        The game played in fictitious play model

    gain : scalar(float), optional(default=None)
        The gain of fictitous play model. If gain is None, the model becomes
        decreasing gain model. If gain is scalar, the model becomes constant
        constant gain model.

    Attributes
    ----------
    g : NomalFormGame
        The game. See Parameters.

    N : scalar(int)
        The number of players in the model.

    players : Player
        Player instance in the model.

    nums_actions : tuple(int)
        Tuple of the number of actions, one for each player.
    """

    def __init__(self, data, gain=None):
        if isinstance(data, NormalFormGame):
            self.g = data
        else:  # data must be array_like
            payoffs = np.asarray(data)
            self.g = NormalFormGame(payoffs)

        self.N = self.g.N
        self.players = self.g.players
        self.nums_actions = self.g.nums_actions
        self.tie_breaking = 'smallest'

        if gain is None:
            self.step_size = lambda t: 1 / (t+1)  # decreasing gain
        else:
            self.step_size = lambda t: gain  # constant gain

    def _play(self, actions, t, random_state=None):
        brs = np.zeros(self.N, dtype=int)
        for i, player in enumerate(self.players):
            index = [j for j in range(i+1, self.N)]
            index.extend([j for j in range(i)])
            opponent_actions = np.asarray([actions[i] for i in index])
            brs[i] = player.best_response(
                opponent_actions if self.N > 2 else opponent_actions[0],
                tie_breaking=self.tie_breaking)

        for i in range(self.N):
            actions[i][:] *= 1 - self.step_size(t+1)
            actions[i][brs[i]] += self.step_size(t+1)

        return actions

    def play(self, init_actions=None, num_reps=1, t_init=0, random_state=None):
        """
        Return a new action profile which is updated by playing the game
        `num_reps` times.

        Parameters
        ----------
        init_actions : tuple(int), optional(default=None)
            The action profile in the first period. If None, selected randomly.

        num_reps : scalar(int), optional(default=1)
            The number of iterations.

        t_init : scalar(int), optional(default=0)
            The period the game starts.

        Returns
        -------
        tuple(int)
            The action profile after iteration.
        """
        random_state = check_random_state(random_state)
        if init_actions is None:
            init_actions = random_pure_actions(self.nums_actions, random_state)
        actions = [i for i in init_actions]
        for i in range(self.N):
            actions[i] = pure2mixed(self.nums_actions[i], init_actions[i])
        for t in range(num_reps):
            actions = self._play(actions, t+t_init, random_state)
        return actions

    def time_series(self, ts_length, init_actions=None, t_init=0,
                    random_state=None):
        """
        Return the array representing time series of normalized action history.

        Parameters
        ----------
        ts_length : scalar(int)
            The number of iterations.

        init_actions : tuple(int), optional(default=None)
            The action profile in the first period. If None, selected randomly.

        t_init : scalar(int), optional(default=0)
            The period the game starts.

        Returns
        -------
        Array
            The array representing time series of normalized action history.
        """
        random_state = check_random_state(random_state)
        if init_actions is None:
            init_actions = random_pure_actions(self.nums_actions, random_state)
        out = [np.empty((ts_length, self.nums_actions[i]))
               for i in range(self.N)]
        actions = [np.empty(self.nums_actions[i]) for i in range(self.N)]
        for i in range(self.N):
            actions[i] = pure2mixed(self.nums_actions[i], init_actions[i])[:]
        for t in range(ts_length):
            for i in range(self.N):
                out[i][t, :] = actions[i][:]
            actions = self._play(actions, t+t_init, random_state)
        return out


class StochasticFictitiousPlay(FictitiousPlay):
    """
    Class representing the stoochastic fictitous play model.
    Subclass of FictitiousPlay.

    Parameters
    ----------
    data : NormalFormGame or array-like
        The game played in the stochastic fictitious play model.

    distribution : scipy.stats object
        statistical distribution from scipy.stats

    gain : scalar(int), optional(default=None)
        The gain of fictitous play model. If gain is None, the model becomes
        decreasing gain model. If gain is scalar, the model becomes constant
        constant gain model.

    Attributes
    ----------
    See attributes of FictitousPlay.
    """

    def __init__(self, data, distribution, gain=None):
        FictitiousPlay.__init__(self, data, gain)

        self.payoff_perturbation_dist = \
            lambda size, random_state: distribution.rvs(
                                        size=size,
                                        random_state=random_state)

        self.tie_breaking = 'smallest'

    def _play(self, actions, t, random_state=None):
        random_state = check_random_state(random_state)
        brs = np.zeros(self.N, dtype=int)
        for i, player in enumerate(self.players):
            index = list(range(i+1, self.N)) + list(range(i))
            opponent_actions = np.asarray([actions[i] for i in index])
            payoff_perturbation = \
                self.payoff_perturbation_dist(size=self.nums_actions[i],
                                              random_state=random_state)
            brs[i] = player.best_response(
                opponent_actions if self.N > 2 else opponent_actions[0],
                tie_breaking=self.tie_breaking,
                payoff_perturbation=payoff_perturbation)

        for i in range(self.N):
            actions[i][:] *= 1 - self.step_size(t+1)
            actions[i][brs[i]] += self.step_size(t+1)

        return actions
