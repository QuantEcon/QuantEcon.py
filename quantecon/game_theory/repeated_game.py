"""
Tools for repeated game.

"""

import numpy as np
from scipy.spatial import ConvexHull
from numba import njit


class RepeatedGame:
    """
    Class representing an N-player repeated game.

    Parameters
    ----------
    stage_game : NormalFormGame
        The stage game used to create the repeated game.

    delta : scalar(float)
        The common discount rate at which all players discount the future.

    Attributes
    ----------
    sg : NormalFormGame
        The stage game. See Parameters.

    delta : scalar(float)
        See Parameters.

    N : scalar(int)
        The number of players.

    nums_actions : tuple(int)
        Tuple of the numbers of actions, one for each player.
    """
    def __init__(self, stage_game, delta):
        self.sg = stage_game
        self.delta = delta
        self.N = stage_game.N
        self.nums_actions = stage_game.nums_actions

    def equilibrium_payoffs(self, method=None, options=None):
        """
        Compute the set of payoff pairs of all pure-strategy subgame-perfect
        equilibria with public randomization for any repeated two-player games
        with perfect monitoring and discounting.

        Parameters
        ----------
        method : str, optional
            The method for solving the equilibrium payoff set.

        options : dict, optional
            A dictionary of method options. For example, 'abreu_sannikov'
            method accepts the following options:

                tol : scalar(float)
                    Tolerance for convergence checking.
                max_iter : scalar(int)
                    Maximum number of iterations.
                u_init : ndarray(float, ndim=1)
                    The initial guess of threat points.

        Notes
        -----
        Here lists all the implemented methods. The default method
        is 'abreu_sannikov'.

            1. 'abreu_sannikov'
        """
        if method is None:
            method = 'abreu_sannikov'

        if options is None:
            options = {}

        if method in ('abreu_sannikov', 'AS'):
            return _equilibrium_payoffs_abreu_sannikov(self, **options)
        else:
            msg = f"method {method} not supported."
            raise NotImplementedError(msg)


def _equilibrium_payoffs_abreu_sannikov(rpg, tol=1e-12, max_iter=500,
                                        u_init=np.zeros(2)):
    """
    Using 'abreu_sannikov' algorithm to compute the set of payoff pairs
    of all pure-strategy subgame-perfect equilibria with public randomization
    for any repeated two-player games with perfect monitoring and
    discounting, following Abreu and Sannikov (2014).

    Parameters
    ----------
    rpg : RepeatedGame
        Two player repeated game.

    tol : scalar(float), optional(default=1e-12)
        Tolerance for convergence checking.

    max_iter : scalar(int), optional(default=500)
        Maximum number of iterations.

    u_init : ndarray(float, ndim=1), optional(default=np.zeros(2))
        The initial guess of threat points.

    Returns
    -------
    hull : scipy.spatial.ConvexHull
        The convex hull of equilibrium payoff pairs.

    References
    ----------
    .. [1] Abreu, Dilip, and Yuliy Sannikov. "An algorithm for
       two‐player repeated games with perfect monitoring." Theoretical
       Economics 9.2 (2014): 313-338.
    """
    sg, delta = rpg.sg, rpg.delta

    if sg.N != 2:
        msg = "this algorithm only applies to repeated two-player games."
        raise NotImplementedError(msg)

    best_dev_gains = _best_dev_gains(rpg)
    IC = np.empty(2)
    action_profile_payoff = np.empty(2)
    # auxiliary array for checking if payoff is inside the convex hull
    # first two entries for payoff point, and the last entry is 1.
    extended_payoff = np.ones(3)
    # array to store new points of C in each intersection
    # at most 4 new points will be generated
    new_pts = np.empty((4, 2))
    # array to store the points of W
    # the length of v is limited by |A1|*|A2|*4
    W_new = np.empty((np.prod(sg.nums_actions)*4, 2))
    W_old = np.empty((np.prod(sg.nums_actions)*4, 2))
    # count the new points generated in each iteration
    n_new_pt = 0

    # copy the threat points
    u = np.copy(u_init)

    # initialization
    payoff_pts = \
        sg.payoff_profile_array.reshape(np.prod(sg.nums_actions), 2)
    W_new[:np.prod(sg.nums_actions)] = payoff_pts
    n_new_pt = np.prod(sg.nums_actions)

    n_iter = 0
    while True:
        W_old[:n_new_pt] = W_new[:n_new_pt]
        n_old_pt = n_new_pt
        hull = ConvexHull(W_old[:n_old_pt])

        W_new, n_new_pt = \
            _R(delta, sg.nums_actions, sg.payoff_arrays,
               best_dev_gains, hull.points, hull.vertices,
               hull.equations, u, IC, action_profile_payoff,
               extended_payoff, new_pts, W_new)

        n_iter += 1
        if n_iter >= max_iter:
            break

        # check convergence
        if n_new_pt == n_old_pt:
            if np.linalg.norm(W_new[:n_new_pt] - W_old[:n_new_pt]) < tol:
                break

        # update threat points
        _update_u(u, W_new[:n_new_pt])

    hull = ConvexHull(W_new[:n_new_pt])

    return hull


def _best_dev_gains(rpg):
    """
    Calculate the normalized payoff gains from deviating from the current
    action to the best response for each player.

    Parameters
    ----------
    rpg : RepeatedGame
        Two player repeated game.

    Returns
    -------
    best_dev_gains : tuple(ndarray(float, ndim=2))
        The normalized best deviation payoff gain arrays.
        best_dev_gains[i][ai, a-i] is normalized payoff gain
        player i can get if originally players are choosing
        ai and a-i, and player i deviates to the best response action.
    """
    sg, delta = rpg.sg, rpg.delta

    best_dev_gains = ((1-delta)/delta *
                      (np.max(sg.payoff_arrays[i], 0) - sg.payoff_arrays[i])
                      for i in range(2))

    return tuple(best_dev_gains)


@njit
def _R(delta, nums_actions, payoff_arrays, best_dev_gains, points,
       vertices, equations, u, IC, action_profile_payoff,
       extended_payoff, new_pts, W_new, tol=1e-10):
    """
    Updating the payoff convex hull by iterating all action pairs.
    Using the R operator proposed by Abreu and Sannikov 2014.

    Parameters
    ----------
    delta : scalar(float)
            The common discount rate at which all players discount
            the future.

    nums_actions : tuple(int)
        Tuple of the numbers of actions, one for each player.

    payoff_arrays : tuple(ndarray(float, ndim=2))
        Tuple of the payoff arrays, one for each player.

    best_dev_gains : tuple(ndarray(float, ndim=2))
        Tuple of the normalized best deviation payoff gain arrays.
        best_dev_gains[i][ai, a-i] is payoff gain player i
        can get if originally players are choosing ai and a-i,
        and player i deviates to the best response action.

    points : ndarray(float, ndim=2)
        Coordinates of the points in the W, which construct a
        feasible payoff convex hull.

    vertices : ndarray(float, ndim=1)
        Indices of points forming the vertices of the convex hull,
        which are in counterclockwise order.

    equations : ndarray(float, ndim=2)
        [normal, offset] forming the hyperplane equation of the facet

    u : ndarray(float, ndim=1)
        The threat points.

    IC : ndarray(float, ndim=1)
        The minimum IC continuation values.

    action_profile_payoff : ndarray(float, ndim=1)
        Array of payoff for one action profile.

    extended_payoff : ndarray(float, ndim=2)
        The array [payoff0, payoff1, 1] for checking if
        [payoff0, payoff1] is in the feasible payoff convex hull.

    new_pts : ndarray(float, ndim=1)
        The 4 by 2 array for storing the generated potential
        extreme points of one action profile. One action profile
        can only generate at most 4 points.

    W_new : ndarray(float, ndim=2)
        Array for storing the coordinates of the generated potential
        extreme points that construct a new feasible payoff convex hull.

    tol: scalar(float), optional(default=1e-10)
        The tolerance for checking if two values are equal.

    Returns
    -------
    W_new : ndarray(float, ndim=2)
        The coordinates of the generated potential extreme points
        that construct a new feasible payoff convex hull.

    n_new_pt : scalar(int)
        The number of points in W_new that construct the feasible
        payoff convex hull.
    """
    n_new_pt = 0
    for a0 in range(nums_actions[0]):
        for a1 in range(nums_actions[1]):
            action_profile_payoff[0] = payoff_arrays[0][a0, a1]
            action_profile_payoff[1] = payoff_arrays[1][a1, a0]
            IC[0] = u[0] + best_dev_gains[0][a0, a1]
            IC[1] = u[1] + best_dev_gains[1][a1, a0]

            # check if payoff is larger than IC
            if (action_profile_payoff >= IC).all():
                # check if payoff is inside the convex hull
                extended_payoff[:2] = action_profile_payoff
                if (np.dot(equations, extended_payoff) <= tol).all():
                    W_new[n_new_pt] = action_profile_payoff
                    n_new_pt += 1
                    continue

            new_pts, n = _find_C(new_pts, points, vertices, equations,
                                 extended_payoff, IC, tol)

            for i in range(n):
                W_new[n_new_pt] = \
                    delta * new_pts[i] + (1-delta) * action_profile_payoff
                n_new_pt += 1

    return W_new, n_new_pt


@njit
def _find_C(C, points, vertices, equations, extended_payoff, IC, tol):
    """
    Find all the intersection points between the current convex hull
    and the two IC constraints. It is done by iterating simplex
    counterclockwise.

    Parameters
    ----------
    C : ndarray(float, ndim=2)
        The 4 by 2 array for storing the generated potential
        extreme points of one action profile. One action profile
        can only generate at most 4 points.

    points : ndarray(float, ndim=2)
        Coordinates of the points in the W, which construct a
        feasible payoff convex hull.

    vertices : ndarray(float, ndim=1)
        Indices of points forming the vertices of the convex hull,
        which are in counterclockwise order.

    equations : ndarray(float, ndim=2)
        [normal, offset] forming the hyperplane equation of the facet

    extended_payoff : ndarray(float, ndim=1)
        The array [payoff0, payoff1, 1] for checking if
        [payoff0, payoff1] is in the feasible payoff convex hull.

    IC : ndarray(float, ndim=1)
        The minimum IC continuation values.

    tol : scalar(float)
        The tolerance for checking if two values are equal.

    Returns
    -------
    C : ndarray(float, ndim=2)
        The generated potential extreme points.

    n : scalar(int)
        The number of found intersection points.
    """
    n = 0
    weights = np.empty(2)
    # vertices is ordered counterclockwise
    for i in range(len(vertices)-1):
        n = _intersect(C, n, weights, IC,
                       points[vertices[i]],
                       points[vertices[i+1]], tol)

    n = _intersect(C, n, weights, IC,
                   points[vertices[-1]],
                   points[vertices[0]], tol)

    # check the case that IC is an interior point of the convex hull
    extended_payoff[:2] = IC
    if (np.dot(equations, extended_payoff) <= tol).all():
        C[n, :] = IC
        n += 1

    return C, n


@njit
def _intersect(C, n, weights, IC, pt0, pt1, tol):
    """
    Find the intersection points of a half-closed simplex
    (pt0, pt1] and IC constraints.

    Parameters
    ----------
    C : ndarray(float, ndim=2)
        The 4 by 2 array for storing the generated points of
        one action profile. One action profile can only
        generate at most 4 points.

    n : scalar(int)
        The number of intersection points that have been found.

    weights : ndarray(float, ndim=1)
        The size 2 array for storing the weights when calculate
        the intersection point of simplex and IC constraints.

    IC : ndarray(float, ndim=1)
        The minimum IC continuation values.

    pt0 : ndarray(float, ndim=1)
        Coordinates of the starting point of the simplex.

    pt1 : ndarray(float, ndim=1)
        Coordinates of the ending point of the simplex.

    tol : scalar(float)
        The tolerance for checking if two values are equal.

    Returns
    -------
    n : scalar(int)
        The updated number of found intersection points.
    """
    for i in range(2):
        if (abs(pt0[i] - pt1[i]) < tol):
            if (abs(pt1[i] - IC[i]) < tol):
                x = pt1[1-i]
            else:
                continue
        else:
            weights[i] = (pt0[i] - IC[i]) / (pt0[i] - pt1[i])
            # pt0 is not included to avoid duplication
            # weights in (0, 1]
            if (0 < weights[i] <= 1):
                x = (1 - weights[i]) * pt0[1-i] + weights[i] * pt1[1-i]
            else:
                continue
        # x has to be strictly higher than IC[1-j]
        # if it is equal, then it means IC is one of the vertex
        # it will be added to C in below
        if x - IC[1-i] > tol:
            C[n, i] = IC[i]
            C[n, 1-i] = x
            n += 1
        elif x - IC[1-i] > -tol:
            # to avoid duplication when IC is a vertex
            break

    return n


@njit
def _update_u(u, W):
    """
    Update the threat points if it not feasible in the new W,
    by the minimum of new feasible payoffs.

    Parameters
    ----------
    u : ndarray(float, ndim=1)
        The threat points.

    W : ndarray(float, ndim=1)
        The points that construct the feasible payoff convex hull.

    Returns
    -------
    u : ndarray(float, ndim=1)
        The updated threat points.
    """
    for i in range(2):
        W_min = W[:, i].min()
        if u[i] < W_min:
            u[i] = W_min

    return u
