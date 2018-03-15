"""
This module contains functions that generate NormalFormGame instances of
the 2-player games studied by Fearnley, Igwe, and Savani (2015):

* Colonel Blotto Games (`blotto_game`): A non-zero sum extension of the
  Blotto game as studied by Hortala-Vallve and Llorente-Saguer (2012),
  where opposing parties have asymmetric and heterogeneous battlefield
  valuations.

* Ranking Games (`ranking_game`): In these games, as studied by Goldberg
  et al. (2013), each player chooses an effort level associated with a
  cost and a score. The players are ranked according to their scores,
  and the player with the higher score wins the prize. Each player's
  payoff is given by the value of the prize minus the cost of the
  effort.

* SGC Games (`sgc_game`): These games were introduced by Sandholm,
  Gilpin, and Conitzer (2005) as a worst case scenario for support
  enumeration as it has a unique equilibrium where each player uses half
  of his actions in his support.

* Tournament Games (`tournament_game`): These games are constructed by
  Anbalagan et al. (2013) as games that do not have interim epsilon-Nash
  equilibria with constant cardinality supports for epsilon smaller than
  a certain threshold.

* Unit Vector Games (`unit_vector_game`): These games are games where
  the payoff matrix of one player consists of unit (column) vectors,
  used by Savani and von Stengel (2016) to construct instances that are
  hard, in terms of computational complexity, both for the Lemke-Howson
  and support enumeration algorithms.

Large part of the code here is based on the C code available at
https://github.com/bimatrix-games/bimatrix-generators distributed under
BSD 3-Clause License.

References
----------
* Y. Anbalagan, S. Norin, R. Savani, and A. Vetta, "Polylogarithmic
  Supports Are Required for Approximate Well-Supported Nash Equilibria
  below 2/3," WINE, 2013.

* J. Fearnley, T. P. Igwe, and R. Savani, "An Empirical Study of Finding
  Approximate Equilibria in Bimatrix Games," International Symposium on
  Experimental Algorithms (SEA), 2015.

* L. A. Goldberg, P. W. Goldberg, P. Krysta, and C. Ventre, "Ranking
  Games that have Competitiveness-based Strategies", Theoretical
  Computer Science, 2013.

* R. Hortala-Vallve and A. Llorente-Saguer, "Pure Strategy Nash
  Equilibria in Non-Zero Sum Colonel Blotto Games", International
  Journal of Game Theory, 2012.

* T. Sandholm, A. Gilpin, and V. Conitzer, "Mixed-Integer Programming
  Methods for Finding Nash Equilibria," AAAI, 2005.

* R. Savani and B. von Stengel, "Unit Vector Games," International
  Journal of Economic Theory, 2016.

"""

# BSD 3-Clause License
#
# Copyright (c) 2015, John Fearnley, Tobenna P. Igwe, Rahul Savani
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import scipy.special
from numba import jit
from ..normal_form_game import Player, NormalFormGame
from ...util import check_random_state
from ...gridtools import simplex_grid
from ...graph_tools import random_tournament_graph
from ...util.combinatorics import next_k_array, k_array_rank_jit


def blotto_game(h, t, rho, mu=0, random_state=None):
    """
    Return a NormalFormGame instance of a 2-player non-zero sum Colonel
    Blotto game (Hortala-Vallve and Llorente-Saguer, 2012), where the
    players have an equal number `t` of troops to assign to `h` hills
    (so that the number of actions for each player is equal to
    (t+h-1) choose (h-1) = (t+h-1)!/(t!*(h-1)!)). Each player has a
    value for each hill that he receives if he assigns strictly more
    troops to the hill than his opponent (ties are broken uniformly at
    random), where the values are drawn from a multivariate normal
    distribution with covariance `rho`. Each player’s payoff is the sum
    of the values of the hills won by that player.

    Parameters
    ----------
    h : scalar(int)
        Number of hills.
    t : scalar(int)
        Number of troops.
    rho : scalar(float)
        Covariance of the players' values of each hill. Must be in
        [-1, 1].
    mu : scalar(float), optional(default=0)
        Mean of the players' values of each hill.
    random_state : int or np.random.RandomState, optional
        Random seed (integer) or np.random.RandomState instance to set
        the initial state of the random number generator for
        reproducibility. If None, a randomly initialized RandomState is
        used.

    Returns
    -------
    g : NormalFormGame

    Examples
    --------
    >>> g = blotto_game(2, 3, 0.5, random_state=1234)
    >>> g.players[0]
    Player([[-0.44861083, -1.08443468, -1.08443468, -1.08443468],
            [ 0.18721302, -0.44861083, -1.08443468, -1.08443468],
            [ 0.18721302,  0.18721302, -0.44861083, -1.08443468],
            [ 0.18721302,  0.18721302,  0.18721302, -0.44861083]])
    >>> g.players[1]
    Player([[-1.20042463, -1.39708658, -1.39708658, -1.39708658],
            [-1.00376268, -1.20042463, -1.39708658, -1.39708658],
            [-1.00376268, -1.00376268, -1.20042463, -1.39708658],
            [-1.00376268, -1.00376268, -1.00376268, -1.20042463]])

    """
    actions = simplex_grid(h, t)
    n = actions.shape[0]
    payoff_arrays = tuple(np.empty((n, n)) for i in range(2))
    mean = np.array([mu, mu])
    cov = np.array([[1, rho], [rho, 1]])
    random_state = check_random_state(random_state)
    values = random_state.multivariate_normal(mean, cov, h)
    _populate_blotto_payoff_arrays(payoff_arrays, actions, values)
    g = NormalFormGame(
        [Player(payoff_array) for payoff_array in payoff_arrays]
    )
    return g


@jit(nopython=True)
def _populate_blotto_payoff_arrays(payoff_arrays, actions, values):
    """
    Populate the ndarrays in `payoff_arrays` with the payoff values of
    the Blotto game with h hills and t troops.

    Parameters
    ----------
    payoff_arrays : tuple(ndarray(float, ndim=2))
        Tuple of 2 ndarrays of shape (n, n), where n = (t+h-1)!/
        (t!*(h-1)!). Modified in place.
    actions : ndarray(int, ndim=2)
        ndarray of shape (n, h) containing all possible actions, i.e.,
        h-part compositions of t.
    values : ndarray(float, ndim=2)
        ndarray of shape (h, 2), where `values[k, :]` contains the
        players' values of hill `k`.

    """
    n, h = actions.shape
    payoffs = np.empty(2)
    for i in range(n):
        for j in range(n):
            payoffs[:] = 0
            for k in range(h):
                if actions[i, k] == actions[j, k]:
                    for p in range(2):
                        payoffs[p] += values[k, p] / 2
                else:
                    winner = np.int(actions[i, k] < actions[j, k])
                    payoffs[winner] += values[k, winner]
            payoff_arrays[0][i, j], payoff_arrays[1][j, i] = payoffs


def ranking_game(n, steps=10, random_state=None):
    """
    Return a NormalFormGame instance of (the 2-player version of) the
    "ranking game" studied by Goldberg et al. (2013), where each player
    chooses an effort level associated with a score and a cost which are
    both increasing functions with randomly generated step sizes. The
    player with the higher score wins the first prize, whose value is 1,
    and the other player obtains the "second prize" of value 0; in the
    case of a tie, the first prize is split and each player receives a
    value of 0.5. The payoff of a player is given by the value of the
    prize minus the cost of the effort.

    Parameters
    ----------
    n : scalar(int)
        Number of actions, i.e, number of possible effort levels.
    steps : scalar(int), optional(default=10)
        Parameter determining the upper bound for the size of the random
        steps for the scores and costs for each player: The step sizes
        for the scores are drawn from `1`, ..., `steps`, while those for
        the costs are multiples of `1/(n*steps)`, where the cost of
        effort level `0` is 0, and the maximum possible cost of effort
        level `n-1` is less than or equal to 1.
    random_state : int or np.random.RandomState, optional
        Random seed (integer) or np.random.RandomState instance to set
        the initial state of the random number generator for
        reproducibility. If None, a randomly initialized RandomState is
        used.

    Returns
    -------
    g : NormalFormGame

    Examples
    --------
    >>> g = ranking_game(5, random_state=1234)
    >>> g.players[0]
    Player([[ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
            [ 0.82, -0.18, -0.18, -0.18, -0.18],
            [ 0.8 ,  0.8 , -0.2 , -0.2 , -0.2 ],
            [ 0.68,  0.68,  0.68, -0.32, -0.32],
            [ 0.66,  0.66,  0.66,  0.66, -0.34]])
    >>> g.players[1]
    Player([[ 1.  ,  0.  ,  0.  ,  0.  ,  0.  ],
            [ 0.8 ,  0.8 , -0.2 , -0.2 , -0.2 ],
            [ 0.66,  0.66,  0.66, -0.34, -0.34],
            [ 0.6 ,  0.6 ,  0.6 ,  0.6 , -0.4 ],
            [ 0.58,  0.58,  0.58,  0.58,  0.58]])

    """
    payoff_arrays = tuple(np.empty((n, n)) for i in range(2))
    random_state = check_random_state(random_state)

    scores = random_state.randint(1, steps+1, size=(2, n))
    scores.cumsum(axis=1, out=scores)

    costs = np.empty((2, n-1))
    costs[:] = random_state.randint(1, steps+1, size=(2, n-1))
    costs.cumsum(axis=1, out=costs)
    costs[:] /= (n * steps)

    _populate_ranking_payoff_arrays(payoff_arrays, scores, costs)
    g = NormalFormGame(
        [Player(payoff_array) for payoff_array in payoff_arrays]
    )
    return g


@jit(nopython=True)
def _populate_ranking_payoff_arrays(payoff_arrays, scores, costs):
    """
    Populate the ndarrays in `payoff_arrays` with the payoff values of
    the ranking game given `scores` and `costs`.

    Parameters
    ----------
    payoff_arrays : tuple(ndarray(float, ndim=2))
        Tuple of 2 ndarrays of shape (n, n). Modified in place.
    scores : ndarray(int, ndim=2)
        ndarray of shape (2, n) containing score values corresponding to
        the effort levels for the two players.
    costs : ndarray(float, ndim=2)
        ndarray of shape (2, n-1) containing cost values corresponding
        to the n-1 positive effort levels for the two players, with the
        assumption that the cost of the zero effort action is zero.

    """
    n = payoff_arrays[0].shape[0]
    for p, payoff_array in enumerate(payoff_arrays):
        payoff_array[0, :] = 0
        for i in range(1, n):
            for j in range(n):
                payoff_array[i, j] = -costs[p, i-1]

    prize = 1.
    for i in range(n):
        for j in range(n):
            if scores[0, i] > scores[1, j]:
                payoff_arrays[0][i, j] += prize
            elif scores[0, i] < scores[1, j]:
                payoff_arrays[1][j, i] += prize
            else:
                payoff_arrays[0][i, j] += prize / 2
                payoff_arrays[1][j, i] += prize / 2


def sgc_game(k):
    """
    Return a NormalFormGame instance of the 2-player game introduced by
    Sandholm, Gilpin, and Conitzer (2005), which has a unique Nash
    equilibrium, where each player plays half of the actions with
    positive probabilities. Payoffs are normalized so that the minimum
    and the maximum payoffs are 0 and 1, respectively.

    Parameters
    ----------
    k : scalar(int)
        Positive integer determining the number of actions. The returned
        game will have `4*k-1` actions for each player.

    Returns
    -------
    g : NormalFormGame

    Examples
    --------
    >>> g = sgc_game(2)
    >>> g.players[0]
    Player([[ 0.75,  0.5 ,  1.  ,  0.5 ,  0.5 ,  0.5 ,  0.5 ],
            [ 1.  ,  0.75,  0.5 ,  0.5 ,  0.5 ,  0.5 ,  0.5 ],
            [ 0.5 ,  1.  ,  0.75,  0.5 ,  0.5 ,  0.5 ,  0.5 ],
            [ 0.  ,  0.  ,  0.  ,  0.75,  0.  ,  0.  ,  0.  ],
            [ 0.  ,  0.  ,  0.  ,  0.  ,  0.75,  0.  ,  0.  ],
            [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.75,  0.  ],
            [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.75]])
    >>> g.players[1]
    Player([[ 0.75,  0.5 ,  1.  ,  0.5 ,  0.5 ,  0.5 ,  0.5 ],
            [ 1.  ,  0.75,  0.5 ,  0.5 ,  0.5 ,  0.5 ,  0.5 ],
            [ 0.5 ,  1.  ,  0.75,  0.5 ,  0.5 ,  0.5 ,  0.5 ],
            [ 0.  ,  0.  ,  0.  ,  0.  ,  0.75,  0.  ,  0.  ],
            [ 0.  ,  0.  ,  0.  ,  0.75,  0.  ,  0.  ,  0.  ],
            [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.75],
            [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.75,  0.  ]])

    """
    payoff_arrays = tuple(np.empty((4*k-1, 4*k-1)) for i in range(2))
    _populate_sgc_payoff_arrays(payoff_arrays)
    g = NormalFormGame(
        [Player(payoff_array) for payoff_array in payoff_arrays]
    )
    return g


@jit(nopython=True)
def _populate_sgc_payoff_arrays(payoff_arrays):
    """
    Populate the ndarrays in `payoff_arrays` with the payoff values of
    the SGC game.

    Parameters
    ----------
    payoff_arrays : tuple(ndarray(float, ndim=2))
        Tuple of 2 ndarrays of shape (4*k-1, 4*k-1). Modified in place.

    """
    n = payoff_arrays[0].shape[0]  # 4*k-1
    m = (n+1)//2 - 1  # 2*k-1
    for payoff_array in payoff_arrays:
        for i in range(m):
            for j in range(m):
                payoff_array[i, j] = 0.75
            for j in range(m, n):
                payoff_array[i, j] = 0.5
        for i in range(m, n):
            for j in range(n):
                payoff_array[i, j] = 0

        payoff_array[0, m-1] = 1
        payoff_array[0, 1] = 0.5
        for i in range(1, m-1):
            payoff_array[i, i-1] = 1
            payoff_array[i, i+1] = 0.5
        payoff_array[m-1, m-2] = 1
        payoff_array[m-1, 0] = 0.5

    k = (m+1)//2
    for h in range(k):
        i, j = m + 2*h, m + 2*h
        payoff_arrays[0][i, j] = 0.75
        payoff_arrays[0][i+1, j+1] = 0.75
        payoff_arrays[1][j, i+1] = 0.75
        payoff_arrays[1][j+1, i] = 0.75


def tournament_game(n, k, random_state=None):
    """
    Return a NormalFormGame instance of the 2-player win-lose game,
    whose payoffs are either 0 or 1, introduced by Anbalagan et al.
    (2013). Player 0 has n actions, which constitute the set of nodes
    {0, ..., n-1}, while player 1 has n choose k actions, each
    corresponding to a subset of k elements of the set of n nodes. Given
    a randomly generated tournament graph on the n nodes, the payoff for
    player 0 is 1 if, in the tournament, the node chosen by player 0
    dominates all the nodes in the k-subset chosen by player 1. The
    payoff for player 1 is 1 if player 1's k-subset contains player 0's
    chosen node.

    Parameters
    ----------
    n : scalar(int)
        Number of nodes in the tournament graph.
    k : scalar(int)
        Size of subsets of nodes in the tournament graph.
    random_state : int or np.random.RandomState, optional
        Random seed (integer) or np.random.RandomState instance to set
        the initial state of the random number generator for
        reproducibility. If None, a randomly initialized RandomState is
        used.

    Returns
    -------
    g : NormalFormGame

    Notes
    -----
    The actions of player 1 are ordered according to the combinatorial
    number system [1]_, which is different from the order used in the
    original library in C.

    Examples
    --------
    >>> g = tournament_game(5, 2, random_state=1234)
    >>> g.players[0]
    Player([[ 0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.],
            [ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
            [ 0.,  1.,  0.,  1.,  0.,  1.,  0.,  0.,  0.,  0.]])
    >>> g.players[1]
    Player([[ 1.,  1.,  0.,  0.,  0.],
            [ 1.,  0.,  1.,  0.,  0.],
            [ 0.,  1.,  1.,  0.,  0.],
            [ 1.,  0.,  0.,  1.,  0.],
            [ 0.,  1.,  0.,  1.,  0.],
            [ 0.,  0.,  1.,  1.,  0.],
            [ 1.,  0.,  0.,  0.,  1.],
            [ 0.,  1.,  0.,  0.,  1.],
            [ 0.,  0.,  1.,  0.,  1.],
            [ 0.,  0.,  0.,  1.,  1.]])

    References
    ----------
    .. [1] `Combinatorial number system
       <https://en.wikipedia.org/wiki/Combinatorial_number_system>`_,
       Wikipedia.

    """
    m = scipy.special.comb(n, k, exact=True)
    if m > np.iinfo(np.intp).max:
        raise ValueError('Maximum allowed size exceeded')

    payoff_arrays = tuple(np.zeros(shape) for shape in [(n, m), (m, n)])
    tourn = random_tournament_graph(n, random_state=random_state)
    indices, indptr = tourn.csgraph.indices, tourn.csgraph.indptr
    _populate_tournament_payoff_array0(payoff_arrays[0], k, indices, indptr)
    _populate_tournament_payoff_array1(payoff_arrays[1], k)
    g = NormalFormGame(
        [Player(payoff_array) for payoff_array in payoff_arrays]
    )
    return g


@jit(nopython=True)
def _populate_tournament_payoff_array0(payoff_array, k, indices, indptr):
    """
    Populate `payoff_array` with the payoff values for player 0 in the
    tournament game given a random tournament graph in CSR format.

    Parameters
    ----------
    payoff_array : ndarray(float, ndim=2)
        ndarray of shape (n, m), where m = n choose k, prefilled with
        zeros. Modified in place.
    k : scalar(int)
        Size of the subsets of nodes.
    indices : ndarray(int, ndim=1)
        CSR format index array of the adjacency matrix of the tournament
        graph.
    indptr : ndarray(int, ndim=1)
        CSR format index pointer array of the adjacency matrix of the
        tournament graph.

    """
    n = payoff_array.shape[0]
    X = np.empty(k, dtype=np.int_)
    a = np.empty(k, dtype=np.int_)
    for i in range(n):
        d = indptr[i+1] - indptr[i]
        if d >= k:
            for j in range(k):
                a[j] = j
            while a[-1] < d:
                for j in range(k):
                    X[j] = indices[indptr[i]+a[j]]
                payoff_array[i, k_array_rank_jit(X)] = 1
                a = next_k_array(a)


@jit(nopython=True)
def _populate_tournament_payoff_array1(payoff_array, k):
    """
    Populate `payoff_array` with the payoff values for player 1 in the
    tournament game.

    Parameters
    ----------
    payoff_array : ndarray(float, ndim=2)
        ndarray of shape (m, n), where m = n choose k, prefilled with
        zeros. Modified in place.
    k : scalar(int)
        Size of the subsets of nodes.

    """
    m = payoff_array.shape[0]
    X = np.arange(k)
    for j in range(m):
        for i in range(k):
            payoff_array[j, X[i]] = 1
        X = next_k_array(X)


def unit_vector_game(n, avoid_pure_nash=False, random_state=None):
    """
    Return a NormalFormGame instance of the 2-player game "unit vector
    game" (Savani and von Stengel, 2016). Payoffs for player 1 are
    chosen randomly from the [0, 1) range. For player 0, each column
    contains exactly one 1 payoff and the rest is 0.

    Parameters
    ----------
    n : scalar(int)
        Number of actions.
    avoid_pure_nash : bool, optional(default=False)
        If True, player 0's payoffs will be placed in order to avoid
        pure Nash equilibria. (If necessary, the payoffs for player 1
        are redrawn so as not to have a dominant action.)
    random_state : int or np.random.RandomState, optional
        Random seed (integer) or np.random.RandomState instance to set
        the initial state of the random number generator for
        reproducibility. If None, a randomly initialized RandomState is
        used.

    Returns
    -------
    g : NormalFormGame

    Examples
    --------
    >>> g = unit_vector_game(4, random_state=1234)
    >>> g.players[0]
    Player([[ 1.,  0.,  1.,  0.],
            [ 0.,  0.,  0.,  1.],
            [ 0.,  0.,  0.,  0.],
            [ 0.,  1.,  0.,  0.]])
    >>> g.players[1]
    Player([[ 0.19151945,  0.62210877,  0.43772774,  0.78535858],
            [ 0.77997581,  0.27259261,  0.27646426,  0.80187218],
            [ 0.95813935,  0.87593263,  0.35781727,  0.50099513],
            [ 0.68346294,  0.71270203,  0.37025075,  0.56119619]])

    With `avoid_pure_nash=True`:

    >>> g = unit_vector_game(4, avoid_pure_nash=True, random_state=1234)
    >>> g.players[0]
    Player([[ 1.,  1.,  0.,  0.],
            [ 0.,  0.,  0.,  0.],
            [ 0.,  0.,  1.,  1.],
            [ 0.,  0.,  0.,  0.]])
    >>> g.players[1]
    Player([[ 0.19151945,  0.62210877,  0.43772774,  0.78535858],
            [ 0.77997581,  0.27259261,  0.27646426,  0.80187218],
            [ 0.95813935,  0.87593263,  0.35781727,  0.50099513],
            [ 0.68346294,  0.71270203,  0.37025075,  0.56119619]])
    >>> pure_nash_brute(g)
    []

    """
    random_state = check_random_state(random_state)
    payoff_arrays = (np.zeros((n, n)), random_state.random_sample((n, n)))

    if not avoid_pure_nash:
        ones_ind = random_state.randint(n, size=n)
        payoff_arrays[0][ones_ind, np.arange(n)] = 1
    else:
        if n == 1:
            raise ValueError('Cannot avoid pure Nash with n=1')
        maxes = payoff_arrays[1].max(axis=0)
        is_suboptimal = payoff_arrays[1] < maxes
        nums_suboptimal = is_suboptimal.sum(axis=1)

        while (nums_suboptimal==0).any():
            payoff_arrays[1][:] = random_state.random_sample((n, n))
            payoff_arrays[1].max(axis=0, out=maxes)
            np.less(payoff_arrays[1], maxes, out=is_suboptimal)
            is_suboptimal.sum(axis=1, out=nums_suboptimal)

        for i in range(n):
            one_ind = random_state.randint(n)
            while not is_suboptimal[i, one_ind]:
                one_ind = random_state.randint(n)
            payoff_arrays[0][one_ind, i] = 1

    g = NormalFormGame(
        [Player(payoff_array) for payoff_array in payoff_arrays]
    )
    return g
