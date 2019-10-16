r"""
Tools for normal form games.

Definitions and Basic Concepts
------------------------------

An :math:`N`-player *normal form game* :math:`g = (I, (A_i)_{i \in I},
(u_i)_{i \in I})` consists of

- the set of *players* :math:`I = \{0, \ldots, N-1\}`,
- the set of *actions* :math:`A_i = \{0, \ldots, n_i-1\}` for each
  player :math:`i \in I`, and
- the *payoff function* :math:`u_i \colon A_i \times A_{i+1} \times
  \cdots \times A_{i+N-1} \to \mathbb{R}` for each player :math:`i \in
  I`,

where :math:`i+j` is understood modulo :math:`N`. Note that we adopt the
convention that the 0-th argument of the payoff function :math:`u_i` is
player :math:`i`'s own action and the :math:`j`-th argument is player
(:math:`i+j`)'s action (modulo :math:`N`). A mixed action for player
:math:`i` is a probability distribution on :math:`A_i` (while an element
of :math:`A_i` is referred to as a pure action). A pure action
:math:`a_i \in A_i` is identified with the mixed action that assigns
probability one to :math:`a_i`. Denote the set of mixed actions of
player :math:`i` by :math:`X_i`. We also denote :math:`A_{-i} = A_{i+1}
\times \cdots \times A_{i+N-1}` and :math:`X_{-i} = X_{i+1} \times
\cdots \times X_{i+N-1}`.

The (pure-action) *best response correspondence* :math:`b_i \colon
X_{-i} \to A_i` for each player :math:`i` is defined by

.. math::

    b_i(x_{-i}) = \{a_i \in A_i \mid
        u_i(a_i, x_{-i}) \geq u_i(a_i', x_{-i})
        \ \forall\,a_i' \in A_i\},

where :math:`u_i(a_i, x_{-i}) = \sum_{a_{-i} \in A_{-i}} u_i(a_i,
a_{-i}) \prod_{j=1}^{N-1} x_{i+j}(a_j)` is the expected payoff to action
:math:`a_i` against mixed actions :math:`x_{-i}`. A profile of mixed
actions :math:`x^* \in X_0 \times \cdots \times X_{N-1}` is a *Nash
equilibrium* if for all :math:`i \in I` and :math:`a_i \in A_i`,

.. math::

    x_i^*(a_i) > 0 \Rightarrow a_i \in b_i(x_{-i}^*),

or equivalently, :math:`x_i^* \cdot v_i(x_{-i}^*) \geq x_i \cdot
v_i(x_{-i}^*)` for all :math:`x_i \in X_i`, where :math:`v_i(x_{-i})` is
the vector of player :math:`i`'s payoffs when the opponent players play
mixed actions :math:`x_{-i}`.

Creating a NormalFormGame
-------------------------

There are three ways to construct a `NormalFormGame` instance.

The first is to pass an array of payoffs for all the players:

>>> matching_pennies_bimatrix = [[(1, -1), (-1, 1)], [(-1, 1), (1, -1)]]
>>> g = NormalFormGame(matching_pennies_bimatrix)
>>> print(g.players[0])
Player in a 2-player normal form game with payoff array:
[[ 1, -1],
 [-1,  1]]
>>> print(g.players[1])
Player in a 2-player normal form game with payoff array:
[[-1,  1],
 [ 1, -1]]

If a square matrix (2-dimensional array) is given, then it is considered
to be a symmetric two-player game:

>>> coordination_game_matrix = [[4, 0], [3, 2]]
>>> g = NormalFormGame(coordination_game_matrix)
>>> print(g)
2-player NormalFormGame with payoff profile array:
[[[4, 4],  [0, 3]],
 [[3, 0],  [2, 2]]]

The second is to specify the sizes of the action sets of the players,
which gives a `NormalFormGame` instance filled with payoff zeros, and
then set the payoff values to each entry:

>>> g = NormalFormGame((2, 2))
>>> print(g)
2-player NormalFormGame with payoff profile array:
[[[ 0.,  0.],  [ 0.,  0.]],
 [[ 0.,  0.],  [ 0.,  0.]]]
>>> g[0, 0] = 1, 1
>>> g[0, 1] = -2, 3
>>> g[1, 0] = 3, -2
>>> print(g)
2-player NormalFormGame with payoff profile array:
[[[ 1.,  1.],  [-2.,  3.]],
 [[ 3., -2.],  [ 0.,  0.]]]

The third is to pass an array of `Player` instances, as explained in the
next section.

Creating a Player
-----------------

A `Player` instance is created by passing a payoff array:

>>> player0 = Player([[3, 1], [0, 2]])
>>> player0.payoff_array
array([[3, 1],
       [0, 2]])

Passing an array of `Player` instances is the third way to create a
`NormalFormGame` instance.

>>> player1 = Player([[2, 0], [1, 3]])
>>> player1.payoff_array
array([[2, 0],
       [1, 3]])
>>> g = NormalFormGame((player0, player1))
>>> print(g)
2-player NormalFormGame with payoff profile array:
[[[3, 2],  [1, 1]],
 [[0, 0],  [2, 3]]]

Beware that in `payoff_array[h, k]`, `h` refers to the player's own
action, while `k` refers to the opponent player's action.

"""
import re
import numbers
import numpy as np
from numba import jit

from ..util import check_random_state


class Player:
    """
    Class representing a player in an N-player normal form game.

    Parameters
    ----------
    payoff_array : array_like(float)
        Array representing the player's payoff function, where
        `payoff_array[a_0, a_1, ..., a_{N-1}]` is the payoff to the
        player when the player plays action `a_0` while his N-1
        opponents play actions `a_1`, ..., `a_{N-1}`, respectively.

    Attributes
    ----------
    payoff_array : ndarray(float, ndim=N)
        See Parameters.

    num_actions : scalar(int)
        The number of actions available to the player.

    num_opponents : scalar(int)
        The number of opponent players.

    dtype : dtype
        Data type of the elements of `payoff_array`.

    tol : scalar(float), default=1e-8
        Default tolerance value used in determining best responses.

    """
    def __init__(self, payoff_array):
        self.payoff_array = np.asarray(payoff_array, order='C')

        if self.payoff_array.ndim == 0:
            raise ValueError('payoff_array must be an array_like')
        if np.prod(self.payoff_array.shape) == 0:
            raise ValueError('every player must have at least one action')

        self.num_opponents = self.payoff_array.ndim - 1
        self.num_actions = self.payoff_array.shape[0]
        self.dtype = self.payoff_array.dtype

        self.tol = 1e-8

    def __repr__(self):
        # From numpy.matrix.__repr__
        # Print also dtype, except for int64, float64
        s = repr(self.payoff_array).replace('array', 'Player')
        l = s.splitlines()
        for i in range(1, len(l)):
            if l[i]:
                l[i] = ' ' + l[i]
        return '\n'.join(l)

    def __str__(self):
        N = self.num_opponents + 1
        s = 'Player in a {N}-player normal form game'.format(N=N)
        s += ' with payoff array:\n'
        s += np.array2string(self.payoff_array, separator=', ')
        return s

    def delete_action(self, action, player_idx=0):
        """
        Return a new `Player` instance with the action(s) specified by
        `action` deleted from the action set of the player specified by
        `player_idx`. Deletion is not performed in place.

        Parameters
        ----------
        action : scalar(int) or array_like(int)
            Integer or array like of integers representing the action(s)
            to be deleted.

        player_idx : scalar(int), optional(default=0)
            Index of the player to delete action(s) for.

        Returns
        -------
        Player
            Copy of `self` with the action(s) deleted as specified.

        Examples
        --------
        >>> player = Player([[3, 0], [0, 3], [1, 1]])
        >>> player
        Player([[3, 0],
                [0, 3],
                [1, 1]])
        >>> player.delete_action(2)
        Player([[3, 0],
                [0, 3]])
        >>> player.delete_action(0, player_idx=1)
        Player([[0],
                [3],
                [1]])

        """
        payoff_array_new = np.delete(self.payoff_array, action, player_idx)
        return Player(payoff_array_new)

    def payoff_vector(self, opponents_actions):
        """
        Return an array of payoff values, one for each own action, given
        a profile of the opponents' actions.

        Parameters
        ----------
        opponents_actions : see `best_response`.

        Returns
        -------
        payoff_vector : ndarray(float, ndim=1)
            An array representing the player's payoff vector given the
            profile of the opponents' actions.

        """
        def reduce_last_player(payoff_array, action):
            """
            Given `payoff_array` with ndim=M, return the payoff array
            with ndim=M-1 fixing the last player's action to be `action`.

            """
            if isinstance(action, numbers.Integral):  # pure action
                return payoff_array.take(action, axis=-1)
            else:  # mixed action
                return payoff_array.dot(action)

        if self.num_opponents == 1:
            payoff_vector = \
                reduce_last_player(self.payoff_array, opponents_actions)
        elif self.num_opponents >= 2:
            payoff_vector = self.payoff_array
            for i in reversed(range(self.num_opponents)):
                payoff_vector = \
                    reduce_last_player(payoff_vector, opponents_actions[i])
        else:  # Trivial case with self.num_opponents == 0
            payoff_vector = self.payoff_array

        return payoff_vector

    def is_best_response(self, own_action, opponents_actions, tol=None):
        """
        Return True if `own_action` is a best response to
        `opponents_actions`.

        Parameters
        ----------
        own_action : scalar(int) or array_like(float, ndim=1)
            An integer representing a pure action, or an array of floats
            representing a mixed action.

        opponents_actions : see `best_response`

        tol : scalar(float), optional(default=None)
            Tolerance level used in determining best responses. If None,
            default to the value of the `tol` attribute.

        Returns
        -------
        bool
            True if `own_action` is a best response to
            `opponents_actions`; False otherwise.

        """
        if tol is None:
            tol = self.tol

        payoff_vector = self.payoff_vector(opponents_actions)
        payoff_max = payoff_vector.max()

        if isinstance(own_action, numbers.Integral):
            return payoff_vector[own_action] >= payoff_max - tol
        else:
            return np.dot(own_action, payoff_vector) >= payoff_max - tol

    def best_response(self, opponents_actions, tie_breaking='smallest',
                      payoff_perturbation=None, tol=None, random_state=None):
        """
        Return the best response action(s) to `opponents_actions`.

        Parameters
        ----------
        opponents_actions : scalar(int) or array_like
            A profile of N-1 opponents' actions, represented by either
            scalar(int), array_like(float), array_like(int), or
            array_like(array_like(float)). If N=2, then it must be a
            scalar of integer (in which case it is treated as the
            opponent's pure action) or a 1-dimensional array of floats
            (in which case it is treated as the opponent's mixed
            action). If N>2, then it must be an array of N-1 objects,
            where each object must be an integer (pure action) or an
            array of floats (mixed action).

        tie_breaking : str, optional(default='smallest')
            str in {'smallest', 'random', False}. Control how, or
            whether, to break a tie (see Returns for details).

        payoff_perturbation : array_like(float), optional(default=None)
            Array of length equal to the number of actions of the player
            containing the values ("noises") to be added to the payoffs
            in determining the best response.

        tol : scalar(float), optional(default=None)
            Tolerance level used in determining best responses. If None,
            default to the value of the `tol` attribute.

        random_state : int or np.random.RandomState, optional
            Random seed (integer) or np.random.RandomState instance to
            set the initial state of the random number generator for
            reproducibility. If None, a randomly initialized RandomState
            is used. Relevant only when tie_breaking='random'.

        Returns
        -------
        scalar(int) or ndarray(int, ndim=1)
            If tie_breaking=False, returns an array containing all the
            best response pure actions. If tie_breaking='smallest',
            returns the best response action with the smallest index; if
            tie_breaking='random', returns an action randomly chosen
            from the best response actions.

        """
        if tol is None:
            tol = self.tol

        payoff_vector = self.payoff_vector(opponents_actions)
        if payoff_perturbation is not None:
            try:
                payoff_vector += payoff_perturbation
            except TypeError:  # type mismatch
                payoff_vector = payoff_vector + payoff_perturbation

        best_responses = \
            np.where(payoff_vector >= payoff_vector.max() - tol)[0]

        if tie_breaking == 'smallest':
            return best_responses[0]
        elif tie_breaking == 'random':
            return self.random_choice(best_responses,
                                      random_state=random_state)
        elif tie_breaking is False:
            return best_responses
        else:
            msg = "tie_breaking must be one of 'smallest', 'random', or False"
            raise ValueError(msg)

    def random_choice(self, actions=None, random_state=None):
        """
        Return a pure action chosen randomly from `actions`.

        Parameters
        ----------
        actions : array_like(int), optional(default=None)
            An array of integers representing pure actions.

        random_state : int or np.random.RandomState, optional
            Random seed (integer) or np.random.RandomState instance to
            set the initial state of the random number generator for
            reproducibility. If None, a randomly initialized RandomState
            is used.

        Returns
        -------
        scalar(int)
            If `actions` is given, returns an integer representing a
            pure action chosen randomly from `actions`; if not, an
            action is chosen randomly from the player's all actions.

        """
        random_state = check_random_state(random_state)

        if actions is not None:
            n = len(actions)
        else:
            n = self.num_actions

        if n == 1:
            idx = 0
        else:
            idx = random_state.randint(n)

        if actions is not None:
            return actions[idx]
        else:
            return idx

    def is_dominated(self, action, tol=None, method=None):
        """
        Determine whether `action` is strictly dominated by some mixed
        action.

        Parameters
        ----------
        action : scalar(int)
            Integer representing a pure action.

        tol : scalar(float), optional(default=None)
            Tolerance level used in determining domination. If None,
            default to the value of the `tol` attribute.

        method : str, optional(default=None)
            If None, `lemke_howson` from `quantecon.game_theory` is used
            to solve for a Nash equilibrium of an auxiliary zero-sum
            game. If `method` is set to `'simplex'`, `'interior-point'`,
            or `'revised simplex'`, then `scipy.optimize.linprog` is
            used with the method as specified by `method`.

        Returns
        -------
        bool
            True if `action` is strictly dominated by some mixed action;
            False otherwise.

        """
        if tol is None:
            tol = self.tol

        payoff_array = self.payoff_array

        if self.num_opponents == 0:
            return payoff_array.max() > payoff_array[action] + tol

        ind = np.ones(self.num_actions, dtype=bool)
        ind[action] = False
        D = payoff_array[ind]
        D -= payoff_array[action]
        if D.shape[0] == 0:  # num_actions == 1
            return False
        if self.num_opponents >= 2:
            D.shape = (D.shape[0], np.prod(D.shape[1:]))

        if method is None:
            from .lemke_howson import lemke_howson
            g_zero_sum = NormalFormGame([Player(D), Player(-D.T)])
            NE = lemke_howson(g_zero_sum)
            return NE[0] @ D @ NE[1] > tol
        elif method in ['simplex', 'interior-point', 'revised simplex']:
            from scipy.optimize import linprog
            m, n = D.shape
            A_ub = np.empty((n, m+1))
            A_ub[:, :m] = -D.T
            A_ub[:, -1] = 1  # Slack variable
            b_ub = np.zeros(n)
            A_eq = np.empty((1, m+1))
            A_eq[:, :m] = 1  # Equality constraint
            A_eq[:, -1] = 0
            b_eq = np.ones(1)
            c = np.zeros(m+1)
            c[-1] = -1
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                          method=method)
            if res.success:
                return res.x[-1] > tol
            elif res.status == 2:  # infeasible
                return False
            else:  # pragma: no cover
                msg = 'scipy.optimize.linprog returned {0}'.format(res.status)
                raise RuntimeError(msg)
        else:
            raise ValueError('Unknown method {0}'.format(method))

    def dominated_actions(self, tol=None, method=None):
        """
        Return a list of actions that are strictly dominated by some
        mixed actions.

        Parameters
        ----------
        tol : scalar(float), optional(default=None)
            Tolerance level used in determining domination. If None,
            default to the value of the `tol` attribute.

        method : str, optional(default=None)
            If None, `lemke_howson` from `quantecon.game_theory` is used
            to solve for a Nash equilibrium of an auxiliary zero-sum
            game. If `method` is set to `'simplex'`, `'interior-point'`,
            or `'revised simplex'`, then `scipy.optimize.linprog` is
            used with the method as specified by `method`.

        Returns
        -------
        list(int)
            List of integers representing pure actions, each of which is
            strictly dominated by some mixed action.

        """
        out = []
        for action in range(self.num_actions):
            if self.is_dominated(action, tol=tol, method=method):
                out.append(action)
        return out


class NormalFormGame:
    """
    Class representing an N-player normal form game.

    Parameters
    ----------
    data : array_like of Player, int (ndim=1), or float (ndim=2 or N+1)
        Data to initialize a NormalFormGame. `data` may be an array of
        Players, in which case the shapes of the Players' payoff arrays
        must be consistent. If `data` is an array of N integers, then
        these integers are treated as the numbers of actions of the N
        players and a NormalFormGame is created consisting of payoffs
        all 0 with `data[i]` actions for each player `i`. `data` may
        also be an (N+1)-dimensional array representing payoff profiles.
        If `data` is a square matrix (2-dimensional array), then the
        game will be a symmetric two-player game where the payoff matrix
        of each player is given by the input matrix.

    dtype : data-type, optional(default=None)
        Relevant only when `data` is an array of integers. Data type of
        the players' payoff arrays. If not supplied, default to
        numpy.float64.

    Attributes
    ----------
    players : tuple(Player)
        Tuple of the Player instances of the game.

    N : scalar(int)
        The number of players.

    nums_actions : tuple(int)
        Tuple of the numbers of actions, one for each player.

    payoff_arrays : tuple(ndarray(float, ndim=N))
        Tuple of the payoff arrays, one for each player.

    """
    def __init__(self, data, dtype=None):
        # data represents an array_like of Players
        if hasattr(data, '__getitem__') and isinstance(data[0], Player):
            N = len(data)

            # Check that the shapes of the payoff arrays are consistent
            # and the dtypes coincide
            shape_0 = data[0].payoff_array.shape
            dtype_0 = data[0].payoff_array.dtype
            for i in range(1, N):
                shape = data[i].payoff_array.shape
                if not (
                    len(shape) == N and
                    shape == shape_0[i:] + shape_0[:i]
                ):
                    raise ValueError(
                        'shapes of payoff arrays must be consistent'
                    )
                dtype = data[i].payoff_array.dtype
                if dtype != dtype_0:
                    raise ValueError(
                        'dtypes of payoff arrays must coincide'
                    )

            self.players = tuple(data)
            self.dtype = dtype_0

        # data represents action sizes or a payoff array
        else:
            data = np.asarray(data)

            if data.ndim == 0:  # data represents action size
                # Trivial game consisting of one player
                N = 1
                self.players = (Player(np.zeros(data)),)
                self.dtype = data.dtype

            elif data.ndim == 1:  # data represents action sizes
                N = data.size
                # N instances of Player created
                # with payoff_arrays filled with zeros
                # Payoff values set via __setitem__
                self.players = tuple(
                    Player(np.zeros(tuple(data[i:]) + tuple(data[:i]),
                                    dtype=dtype))
                    for i in range(N)
                )
                self.dtype = self.players[0].payoff_array.dtype

            elif data.ndim == 2 and data.shape[1] >= 2:
                # data represents a payoff array for symmetric two-player game
                # Number of actions must be >= 2
                if data.shape[0] != data.shape[1]:
                    raise ValueError(
                        'symmetric two-player game must be represented ' +
                        'by a square matrix'
                    )
                N = 2
                self.players = tuple(Player(data) for i in range(N))
                self.dtype = data.dtype

            else:  # data represents a payoff array
                # data must be of shape (n_0, ..., n_{N-1}, N),
                # where n_i is the number of actions available to player i,
                # and the last axis contains the payoff profile
                N = data.ndim - 1
                if data.shape[-1] != N:
                    raise ValueError(
                        'size of innermost array must be equal to ' +
                        'the number of players'
                    )
                payoff_arrays = tuple(
                    np.empty(data.shape[i:-1]+data.shape[:i], dtype=data.dtype)
                    for i in range(N)
                )
                for i, payoff_array in enumerate(payoff_arrays):
                    payoff_array[:] = \
                        data.take(i, axis=-1).transpose(list(range(i, N)) +
                                                        list(range(i)))
                self.players = tuple(
                    Player(payoff_array) for payoff_array in payoff_arrays
                )
                self.dtype = data.dtype

        self.N = N  # Number of players
        self.nums_actions = tuple(
            player.num_actions for player in self.players
        )
        self.payoff_arrays = tuple(
            player.payoff_array for player in self.players
        )

    @property
    def payoff_profile_array(self):
        N = self.N
        dtype = self.dtype
        payoff_profile_array = \
            np.empty(self.players[0].payoff_array.shape + (N,), dtype=dtype)
        for i, player in enumerate(self.players):
            payoff_profile_array[..., i] = \
                player.payoff_array.transpose(list(range(N-i, N)) +
                                              list(range(N-i)))
        return payoff_profile_array

    def __repr__(self):
        s = '<{nums_actions} {N}-player NormalFormGame of dtype {dtype}>'
        return s.format(nums_actions=_nums_actions2string(self.nums_actions),
                        N=self.N,
                        dtype=self.dtype)

    def __str__(self):
        s = '{N}-player NormalFormGame with payoff profile array:\n'
        s += _payoff_profile_array2string(self.payoff_profile_array)
        return s.format(N=self.N)

    def __getitem__(self, action_profile):
        if self.N == 1:  # Trivial game with 1 player
            if not isinstance(action_profile, numbers.Integral):
                raise TypeError('index must be an integer')
            return self.players[0].payoff_array[action_profile]

        # Non-trivial game with 2 or more players
        try:
            if len(action_profile) != self.N:
                raise IndexError('index must be of length {0}'.format(self.N))
        except TypeError:
            raise TypeError('index must be a tuple')

        payoff_profile = np.empty(self.N, dtype=self.dtype)
        for i, player in enumerate(self.players):
            payoff_profile[i] = \
                player.payoff_array[
                    tuple(action_profile[i:]) + tuple(action_profile[:i])
                ]

        return payoff_profile

    def __setitem__(self, action_profile, payoff_profile):
        if self.N == 1:  # Trivial game with 1 player
            if not isinstance(action_profile, numbers.Integral):
                raise TypeError('index must be an integer')
            self.players[0].payoff_array[action_profile] = payoff_profile
            return None

        # Non-trivial game with 2 or more players
        try:
            if len(action_profile) != self.N:
                raise IndexError('index must be of length {0}'.format(self.N))
        except TypeError:
            raise TypeError('index must be a tuple')

        try:
            if len(payoff_profile) != self.N:
                raise ValueError(
                    'value must be an array_like of length {0}'.format(self.N)
                )
        except TypeError:
            raise TypeError('value must be a tuple')

        for i, player in enumerate(self.players):
            player.payoff_array[
                tuple(action_profile[i:]) + tuple(action_profile[:i])
            ] = payoff_profile[i]

    def delete_action(self, player_idx, action):
        """
        Return a new `NormalFormGame` instance with the action(s)
        specified by `action` deleted from the action set of the player
        specified by `player_idx`. Deletion is not performed in place.

        Parameters
        ----------
        player_idx : scalar(int)
            Index of the player to delete action(s) for.

        action : scalar(int) or array_like(int)
            Integer or array like of integers representing the action(s)
            to be deleted.

        Returns
        -------
        NormalFormGame
            Copy of `self` with the action(s) deleted as specified.

        Examples
        --------
        >>> g = NormalFormGame(
        ...     [[(3, 0), (0, 1)], [(0, 0), (3, 1)], [(1, 1), (1, 0)]]
        ... )
        >>> print(g)
        2-player NormalFormGame with payoff profile array:
        [[[3, 0],  [0, 1]],
         [[0, 0],  [3, 1]],
         [[1, 1],  [1, 0]]]

        Delete player `0`'s action `2` from `g`:

        >>> g1 = g.delete_action(0, 2)
        >>> print(g1)
        2-player NormalFormGame with payoff profile array:
        [[[3, 0],  [0, 1]],
         [[0, 0],  [3, 1]]]

        Then delete player `1`'s action `0` from `g1`:

        >>> g2 = g1.delete_action(1, 0)
        >>> print(g2)
        2-player NormalFormGame with payoff profile array:
        [[[0, 1]],
         [[3, 1]]]

        """
        # Allow negative indexing
        if -self.N <= player_idx < 0:
            player_idx = player_idx + self.N

        players_new = tuple(
            player.delete_action(action, player_idx-i)
            for i, player in enumerate(self.players)
        )
        return NormalFormGame(players_new)

    def is_nash(self, action_profile, tol=None):
        """
        Return True if `action_profile` is a Nash equilibrium.

        Parameters
        ----------
        action_profile : array_like(int or array_like(float))
            An array of N objects, where each object must be an integer
            (pure action) or an array of floats (mixed action).

        tol : scalar(float)
            Tolerance level used in determining best responses. If None,
            default to each player's `tol` attribute value.

        Returns
        -------
        bool
            True if `action_profile` is a Nash equilibrium; False
            otherwise.

        """
        if self.N == 2:
            for i, player in enumerate(self.players):
                own_action, opponent_action = \
                    action_profile[i], action_profile[1-i]
                if not player.is_best_response(own_action, opponent_action,
                                               tol):
                    return False

        elif self.N >= 3:
            for i, player in enumerate(self.players):
                own_action = action_profile[i]
                opponents_actions = \
                    tuple(action_profile[i+1:]) + tuple(action_profile[:i])

                if not player.is_best_response(own_action, opponents_actions,
                                               tol):
                    return False

        else:  # Trivial case with self.N == 1
            if not self.players[0].is_best_response(action_profile[0], None,
                                                    tol):
                return False

        return True


def _nums_actions2string(nums_actions):
    if len(nums_actions) == 1:
        s = '{0}-action'.format(nums_actions[0])
    else:
        s = 'x'.join(map(str, nums_actions))
    return s


def _payoff_profile_array2string(payoff_profile_array, class_name=None):
    s = np.array2string(payoff_profile_array, separator=', ')

    # Remove one linebreak
    s = re.sub(r'(\n+)', lambda x: x.group(0)[0:-1], s)

    if class_name is not None:
        prefix = class_name + '('
        next_line_prefix = ' ' * len(prefix)
        suffix = ')'
        l = s.splitlines()
        l[0] = prefix + l[0]
        for i in range(1, len(l)):
            if l[i]:
                l[i] = next_line_prefix + l[i]
        l[-1] += suffix
        s = '\n'.join(l)

    return s


def pure2mixed(num_actions, action):
    """
    Convert a pure action to the corresponding mixed action.

    Parameters
    ----------
    num_actions : scalar(int)
        The number of the pure actions (= the length of a mixed action).

    action : scalar(int)
        The pure action to convert to the corresponding mixed action.

    Returns
    -------
    ndarray(float, ndim=1)
        The mixed action representation of the given pure action.

    """
    mixed_action = np.zeros(num_actions)
    mixed_action[action] = 1
    return mixed_action


# Numba jitted functions #

@jit(nopython=True, cache=True)
def best_response_2p(payoff_matrix, opponent_mixed_action, tol=1e-8):
    """
    Numba-optimized version of `Player.best_response` compilied in
    nopython mode, specialized for 2-player games (where there is only
    one opponent).

    Return the best response action (with the smallest index if more
    than one) to `opponent_mixed_action` under `payoff_matrix`.

    Parameters
    ----------
    payoff_matrix : ndarray(float, ndim=2)
        Payoff matrix.

    opponent_mixed_action : ndarray(float, ndim=1)
        Opponent's mixed action. Its length must be equal to
        `payoff_matrix.shape[1]`.

    tol : scalar(float), optional(default=None)
        Tolerance level used in determining best responses.

    Returns
    -------
    scalar(int)
        Best response action.

    """
    n, m = payoff_matrix.shape

    payoff_max = -np.inf
    payoff_vector = np.zeros(n)

    for a in range(n):
        for b in range(m):
            payoff_vector[a] += payoff_matrix[a, b] * opponent_mixed_action[b]
        if payoff_vector[a] > payoff_max:
            payoff_max = payoff_vector[a]

    for a in range(n):
        if payoff_vector[a] >= payoff_max - tol:
            return a
