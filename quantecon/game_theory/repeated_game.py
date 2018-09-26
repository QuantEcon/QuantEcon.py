"""
Tools for repeated game.

"""

import numpy as np
from scipy.spatial import ConvexHull
from numba import jit, njit

class RepeatedGame:
    """
    Class representing an N-player repeated game.

    Parameters
    ----------
    stage_game : NormalFormGame
                 The stage game used to create the repeated game.

    delta : scalar(float)
            The common discount rate at which all players discount the future.

    """
    def __init__(self, stage_game, delta):
        self.sg = stage_game
        self.delta = delta
        self.N = stage_game.N
        self.nums_actions = stage_game.nums_actions

    def AS(self, tol=1e-12, max_iter=500, u=np.zeros(2)):
        """
        Using AS algorithm to compute the set of payoff pairs of all
        pure-strategy subgame-perfect equilibria with public randomization
        for any repeated two-player games with perfect monitoring and
        discounting, following Abreu and Sannikov (2014).

        Parameters
        ----------
        g : NormalFormGame
            NormalFormGame instance with 2 players.

        delta : scalar(float)
            The discounting factor.

        tol : scalar(float), optional(default=1e-12)
            Tolerance for convergence checkinsg.

        max_iter : scalar(int), optional(default=500)
            Maximum number of iterations.

        u : ndarray(float, ndim=1)
            The initial threat points.

        Returns
        -------
        hull : scipy.spatial.ConvexHull
            The convex hull of feasible payoff pairs.
        """
        sg, delta = self.sg, self.delta
        best_dev_gains = _best_dev_gains(sg, delta)
        C = np.empty((4, 2))
        IC = np.empty(2)
        action_profile_payoff = np.empty(2)
        # array for checking if payoff is inside the polytope or not
        # the last entry is set to be 1
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
                R(delta, sg.nums_actions, sg.payoff_arrays,
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
            update_u(u, W_new[:n_new_pt])

        hull = ConvexHull(W_new[:n_new_pt])

        return hull

@jit()
def _best_dev_gains(sg, delta):
    """
    Calculate the normalized payoff gains from deviating from the current
    action to the best response for each player.
    """
    best_dev_gains0 = (1-delta)/delta * \
        (np.max(sg.payoff_arrays[0], 0) - sg.payoff_arrays[0])
    best_dev_gains1 = (1-delta)/delta * \
        (np.max(sg.payoff_arrays[1], 0) - sg.payoff_arrays[1])

    return best_dev_gains0, best_dev_gains1

@njit
def R(delta, nums_actions, payoff_arrays, best_dev_gains, points,
      vertices, equations, u, IC, action_profile_payoff,
      extended_payoff, new_pts, W_new, tol=1e-10):
    """
    Updating the payoff convex hull by iterating all action pairs.
    Using the R operator proposed by Abreu and Sannikov 2014.
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

            new_pts, n = find_C(new_pts, points, vertices, equations,
                                extended_payoff, IC, tol)

            for i in range(n):
                W_new[n_new_pt] = \
                    delta * new_pts[i] + (1-delta) * action_profile_payoff
                n_new_pt += 1

    return W_new, n_new_pt

@njit
def find_C(C, points, vertices, equations, extended_payoff, IC, tol):
    """
    Find all the intersection points between the current polytope
    and the two IC constraints. It is done by iterating simplex
    counterclockwise.
    """
    # record the number of intersections for each IC.
    n = 0
    weights = np.empty(2)
    # vertices is ordered counterclockwise
    for i in range(len(vertices)-1):
        n = intersect(C, n, weights, IC,
                      points[vertices[i]],
                      points[vertices[i+1]], tol)

    n = intersect(C, n, weights, IC,
                  points[vertices[-1]],
                  points[vertices[0]], tol)

    # check the case that IC is a interior point of the polytope
    extended_payoff[:2] = IC
    if (np.dot(equations, extended_payoff) <= tol).all():
        C[n, :] = IC
        n += 1

    return C, n

@njit
def intersect(C, n, weights, IC, pt0, pt1, tol):
    """
    Find the intersection points of a half-closed simplex
    (pt0, pt1] and IC constraints.
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
def update_u(u, v):
    """
    Update the threat points.
    """
    for i in range(2):
        v_min = v[:, i].min()
        if u[i] < v_min:
            u[i] = v_min

    return u
