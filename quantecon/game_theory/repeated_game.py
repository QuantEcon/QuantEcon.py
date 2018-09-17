"""
Algorithms for repeated game.

"""

import numpy as np
from scipy.spatial import ConvexHull
from numba import jit

def AS(g, delta, tol=1e-12, max_iter=500, u=np.zeros(2)):
    """
    Using AS algorithm to compute the set of payoff pairs of all pure-strategy
    subgame-perfect equilibria with public randomization for any repeated
    two-player games with perfect monitoring and discounting, following
    Abreu and Sannikov (2014).

    Parameters
    ----------
    g : NormalFormGame
        NormalFormGame instance with 2 players.

    delta : scalar(float)
        The discounting factor.

    tol : scalar(float), optional(default=1e-12)
        Tolerance for convergence checking.

    max_iter : scalar(int), optional(default=500)
        Maximum number of iterations.

    u : ndarray(float, ndim=1)
        The initial threat points.

    Returns
    -------
    hull : scipy.spatial.ConvexHull
        The convex hull of feasible payoff pairs.
    """
    best_dev_gains = _best_dev_gains(g, delta)
    C = np.empty((4, 2))
    IC = np.empty(2)
    payoff = np.empty(2)
    # array for checking if payoff is inside the polytope or not
    payoff_pts = np.ones(3)
    new_pts = np.empty((4, 2))
    # the length of v is limited by |A1|*|A2|*4
    v_new = np.empty((np.prod(g.nums_actions)*4, 2))
    v_old = np.empty((np.prod(g.nums_actions)*4, 2))
    n_new_pt = 0

    # initialization
    payoff_profile_pts = \
        g.payoff_profile_array.reshape(np.prod(g.nums_actions), 2)
    v_new[:np.prod(g.nums_actions)] = payoff_profile_pts
    n_new_pt = np.prod(g.nums_actions)

    n_iter = 0
    while True:
        v_old[:n_new_pt] = v_new[:n_new_pt]
        n_old_pt = n_new_pt
        hull = ConvexHull(v_old[:n_old_pt])

        v_new, n_new_pt = \
            update_v(delta, g.nums_actions, g.payoff_arrays,
                     best_dev_gains, hull.points, hull.vertices,
                     hull.equations, u, IC, payoff, payoff_pts,
                     new_pts, v_new)
        n_iter += 1
        if n_iter >= max_iter:
            break

        # check convergence
        if n_new_pt == n_old_pt:
            if np.linalg.norm(v_new[:n_new_pt] - v_old[:n_new_pt]) < tol:
                break

        # update threat points
        update_u(u, v_new[:n_new_pt])

    hull = ConvexHull(v_new[:n_new_pt])

    return hull

@jit()
def _best_dev_gains(g, delta):
    """
    Calculate the payoff gains from deviating from the current action to
    the best response for each player.
    """
    best_dev_gains0 = (1-delta)/delta * \
        (np.max(g.payoff_arrays[0], 0) - g.payoff_arrays[0])
    best_dev_gains1 = (1-delta)/delta * \
        (np.max(g.payoff_arrays[1], 0) - g.payoff_arrays[1])

    return best_dev_gains0, best_dev_gains1

@jit(nopython=True)
def update_v(delta, nums_actions, payoff_arrays, best_dev_gains, points,
             vertices, equations, u, IC, payoff, payoff_pts, new_pts,
             v_new, tol=1e-10):
    """
    Updating the payoff convex hull by iterating all action pairs.
    """
    n_new_pt = 0
    for a0 in range(nums_actions[0]):
        for a1 in range(nums_actions[1]):
            payoff[0] = payoff_arrays[0][a0, a1]
            payoff[1] = payoff_arrays[1][a1, a0]
            IC[0] = u[0] + best_dev_gains[0][a0, a1]
            IC[1] = u[1] + best_dev_gains[1][a1, a0]

            # check if payoff is larger than IC
            if (payoff >= IC).all():
                # check if payoff is inside the convex hull
                payoff_pts[:2] = payoff
                if (np.dot(equations, payoff_pts) <= tol).all():
                    v_new[n_new_pt] = payoff
                    n_new_pt += 1
                    continue

            new_pts, n = find_C(new_pts, points, vertices, IC, tol)

            for i in range(n):
                v_new[n_new_pt] = delta * new_pts[i] + (1-delta) * payoff
                n_new_pt += 1

    return v_new, n_new_pt

@jit(nopython=True)
def find_C(C, points, vertices, IC, tol):
    """
    Find all the intersection points between the current polytope
    and the two IC constraints.
    """
    n_IC = [0, 0]
    weights = np.empty(2)
    # vertices is ordered counterclockwise
    for i in range(len(vertices)-1):
        intersect(C, n_IC, weights, IC,
                  points[vertices[i]], points[vertices[i+1]], tol)

    intersect(C, n_IC, weights, IC,
              points[vertices[-1]], points[vertices[0]], tol)

    # check the case that IC is a interior point of the polytope
    n = n_IC[0] + n_IC[1]
    if (n_IC[0] == 1 & n_IC[1] == 1):
        C[2, 0] = IC[0]
        C[2, 1] = IC[1]
        n += 1

    return C, n

@jit(nopython=True)
def intersect(C, n_IC, weights, IC, pt0, pt1, tol):
    """
    Find the intersection points of a simplex and ICs.
    """
    for i in range(2):
        if (abs(pt0[i] - pt1[i]) < tol):
            None
        else:
            weights[i] = (pt0[i] - IC[i]) / (pt0[i] - pt1[i])

            # intersection of IC[j]
            if (0 < weights[i] <= 1):
                x = (1 - weights[i]) * pt0[1-i] + weights[i] * pt1[1-i]
                # x has to be strictly higher than IC[1-j]
                # if it is equal, then it means IC is one of the vertex
                # it will be added to C in below
                if x - IC[1-i] > tol:
                    C[n_IC[0]+n_IC[1], i] = IC[i]
                    C[n_IC[0]+n_IC[1], 1-i] = x
                    n_IC[i] += 1
                elif x - IC[1-i] > -tol:
                    C[n_IC[0]+n_IC[1], i] = IC[i]
                    C[n_IC[0]+n_IC[1], 1-i] = x
                    n_IC[i] += 1
                    # to avoid duplication when IC is a vertex
                    break
    return C, n_IC

@jit(nopython=True)
def update_u(u, v):
    """
    Update the threat points.
    """
    for i in range(2):
        v_min = v[:, i].min()
        if u[i] < v_min:
            u[i] = v_min

    return u
