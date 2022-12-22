"""
Compute mixed Nash equilibria of an N-player normal form game by
applying the imitation game algorithm by McLennan and Tourky to the best
response correspondence.

"""
import numbers
import numpy as np
from ..compute_fp import _compute_fixed_point_ig
from .normal_form_game import pure2mixed
from .utilities import NashResult


def mclennan_tourky(g, init=None, epsilon=1e-3, max_iter=200,
                    full_output=False):
    r"""
    Find one mixed-action epsilon-Nash equilibrium of an N-player normal
    form game by the fixed point computation algorithm by McLennan and
    Tourky [1]_.

    Parameters
    ----------
    g : NormalFormGame
        NormalFormGame instance.

    init : array_like(int or array_like(float, ndim=1)), optional
        Initial action profile, an array of N objects, where each object
        must be an iteger (pure action) or an array of floats (mixed
        action). If None, default to an array of zeros (the zero-th
        action for each player).

    epsilon : scalar(float), optional(default=1e-3)
        Value of epsilon-optimality.

    max_iter : scalar(int), optional(default=100)
        Maximum number of iterations.

    full_output : bool, optional(default=False)
        If False, only the computed Nash equilibrium is returned. If
        True, the return value is `(NE, res)`, where `NE` is the Nash
        equilibrium and `res` is a `NashResult` object.

    Returns
    -------
    NE : tuple(ndarray(float, ndim=1))
        Tuple of computed Nash equilibrium mixed actions.

    res : NashResult
        Object containing information about the computation. Returned
        only when `full_output` is True. See `NashResult` for details.

    Examples
    --------
    Consider the following version of 3-player "anti-coordination" game,
    where action 0 is a safe action which yields payoff 1, while action
    1 yields payoff :math:`v` if no other player plays 1 and payoff 0
    otherwise:

    >>> N = 3
    >>> v = 2
    >>> payoff_array = np.empty((2,)*n)
    >>> payoff_array[0, :] = 1
    >>> payoff_array[1, :] = 0
    >>> payoff_array[1].flat[0] = v
    >>> g = NormalFormGame((Player(payoff_array),)*N)
    >>> print(g)
    3-player NormalFormGame with payoff profile array:
    [[[[ 1.,  1.,  1.],   [ 1.,  1.,  2.]],
      [[ 1.,  2.,  1.],   [ 1.,  0.,  0.]]],
     [[[ 2.,  1.,  1.],   [ 0.,  1.,  0.]],
      [[ 0.,  0.,  1.],   [ 0.,  0.,  0.]]]]

    This game has a unique symmetric Nash equilibrium, where the
    equilibrium action is given by :math:`(p^*, 1-p^*)` with :math:`p^*
    = 1/v^{1/(N-1)}`:

    >>> p_star = 1/(v**(1/(N-1)))
    >>> [p_star, 1 - p_star]
    [0.7071067811865475, 0.29289321881345254]

    Obtain an approximate Nash equilibrium of this game by
    `mclennan_tourky`:

    >>> epsilon = 1e-5  # Value of epsilon-optimality
    >>> NE = mclennan_tourky(g, epsilon=epsilon)
    >>> print(NE[0], NE[1], NE[2], sep='\n')
    [ 0.70710754  0.29289246]
    [ 0.70710754  0.29289246]
    [ 0.70710754  0.29289246]
    >>> g.is_nash(NE, tol=epsilon)
    True

    Additional information is returned if `full_output` is set True:

    >>> NE, res = mclennan_tourky(g, epsilon=epsilon, full_output=True)
    >>> res.converged
    True
    >>> res.num_iter
    18

    References
    ----------
    .. [1] A. McLennan and R. Tourky, "From Imitation Games to
       Kakutani," 2006.

    """
    try:
        N = g.N
    except:
        raise TypeError('g must be a NormalFormGame')
    if N < 2:
        raise NotImplementedError('Not implemented for 1-player games')

    if init is None:
        init = (0,) * N
    try:
        l = len(init)
    except TypeError:
        raise TypeError('init must be array_like')
    if l != N:
        raise ValueError(
            'init must be of length {N}'.format(N=N)
        )

    indptr = np.empty(N+1, dtype=int)
    indptr[0] = 0
    indptr[1:] = np.cumsum(g.nums_actions)
    x_init = _flatten_action_profile(init, indptr)

    is_approx_fp = lambda x: _is_epsilon_nash(x, g, epsilon, indptr)
    x_star, converged, num_iter = \
        _compute_fixed_point_ig(_best_response_selection, x_init, max_iter,
                                verbose=0, print_skip=1,
                                is_approx_fp=is_approx_fp,
                                g=g, indptr=indptr)
    NE = _get_action_profile(x_star, indptr)

    if not full_output:
        return NE

    res = NashResult(NE=NE,
                     converged=converged,
                     num_iter=num_iter,
                     max_iter=max_iter,
                     init=init,
                     epsilon=epsilon)

    return NE, res


def _best_response_selection(x, g, indptr=None):
    """
    Selection of the best response correspondence of `g` that selects
    the best response action with the smallest index when there are
    ties, where the input and output are flattened action profiles.

    Parameters
    ----------
    x : array_like(float, ndim=1)
        Array of flattened mixed action profile of length equal to n_0 +
        ... + n_N-1, where `out[indptr[i]:indptr[i+1]]` contains player
        i's mixed action.

    g : NormalFormGame

    indptr : array_like(int, ndim=1), optional(default=None)
        Array of index pointers of length N+1, where `indptr[0] = 0` and
        `indptr[i+1] = indptr[i] + n_i`. Created internally if None.

    Returns
    -------
    out : ndarray(float, ndim=1)
        Array of flattened mixed action profile of length equal to n_0 +
        ... + n_N-1, where `out[indptr[i]:indptr[i+1]]` contains player
        i's mixed action representation of his pure best response.

    """
    N = g.N

    if indptr is None:
        indptr = np.empty(N+1, dtype=int)
        indptr[0] = 0
        indptr[1:] = np.cumsum(g.nums_actions)

    out = np.zeros(indptr[-1])

    if N == 2:
        for i in range(N):
            opponent_action = x[indptr[1-i]:indptr[1-i+1]]
            pure_br = g.players[i].best_response(opponent_action)
            out[indptr[i]+pure_br] = 1
    else:
        for i in range(N):
            opponent_actions = tuple(
                x[indptr[(i+j)%N]:indptr[(i+j)%N+1]] for j in range(1, N)
            )
            pure_br = g.players[i].best_response(opponent_actions)
            out[indptr[i]+pure_br] = 1

    return out


def _is_epsilon_nash(x, g, epsilon, indptr=None):
    """
    Determine whether `x` is an `epsilon`-Nash equilibrium of `g`.

    Parameters
    ----------
    x : array_like(float, ndim=1)
        Array of flattened mixed action profile of length equal to n_0 +
        ... + n_N-1, where `out[indptr[i]:indptr[i+1]]` contains player
        i's mixed action.

    g : NormalFormGame

    epsilon : scalar(float)

    indptr : array_like(int, ndim=1), optional(default=None)
        Array of index pointers of length N+1, where `indptr[0] = 0` and
        `indptr[i+1] = indptr[i] + n_i`. Created internally if None.

    Returns
    -------
    bool

    """
    if indptr is None:
        indptr = np.empty(g.N+1, dtype=int)
        indptr[0] = 0
        indptr[1:] = np.cumsum(g.nums_actions)

    action_profile = _get_action_profile(x, indptr)
    return g.is_nash(action_profile, tol=epsilon)


def _get_action_profile(x, indptr):
    """
    Obtain a tuple of mixed actions from a flattened action profile.

    Parameters
    ----------
    x : array_like(float, ndim=1)
        Array of flattened mixed action profile of length equal to n_0 +
        ... + n_N-1, where `out[indptr[i]:indptr[i+1]]` contains player
        i's mixed action.

    indptr : array_like(int, ndim=1)
        Array of index pointers of length N+1, where `indptr[0] = 0` and
        `indptr[i+1] = indptr[i] + n_i`.

    Returns
    -------
    action_profile : tuple(ndarray(float, ndim=1))
        Tuple of N mixed actions, each of length n_i.

    """
    N = len(indptr) - 1
    action_profile = tuple(x[indptr[i]:indptr[i+1]] for i in range(N))
    return action_profile


def _flatten_action_profile(action_profile, indptr):
    """
    Flatten the given action profile.

    Parameters
    ----------
    action_profile : array_like(int or array_like(float, ndim=1))
        Profile of actions of the N players, where each player i' action
        is a pure action (int) or a mixed action (array_like of floats
        of length n_i).

    indptr : array_like(int, ndim=1)
        Array of index pointers of length N+1, where `indptr[0] = 0` and
        `indptr[i+1] = indptr[i] + n_i`.

    Returns
    -------
    out : ndarray(float, ndim=1)
        Array of flattened mixed action profile of length equal to n_0 +
        ... + n_N-1, where `out[indptr[i]:indptr[i+1]]` contains player
        i's mixed action.

    """
    N = len(indptr) - 1
    out = np.empty(indptr[-1])

    for i in range(N):
        if isinstance(action_profile[i], numbers.Integral):  # pure action
            num_actions = indptr[i+1] - indptr[i]
            mixed_action = pure2mixed(num_actions, action_profile[i])
        else:  # mixed action
            mixed_action = action_profile[i]
        out[indptr[i]:indptr[i+1]] = mixed_action

    return out
