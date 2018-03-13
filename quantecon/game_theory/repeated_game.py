"""
Filename: repeated_game.py
Authors: Chase Coleman, Quentin Batista

Tools for repeated games.

"""

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import HalfspaceIntersection
from .pure_nash import pure_nash_brute_gen
from ..ce_util import gridmake


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

    def outerapproximation(self, nH=32, tol=1e-8, maxiter=500,
                           check_pure_nash=True, verbose=False, nskipprint=50,
                           linprog_method='simplex'):
        """
        Approximates the set of equilibrium value set for a repeated game with
        the outer hyperplane approximation described by Judd, Yeltekin,
        Conklin 2002.

        Parameters
        ----------
        rpd : RepeatedGame
            Two player repeated game.
        nH : scalar(int), optional(default=32)
            Number of subgradients used for the approximation.
        tol : scalar(float), optional(default=1e-8)
            Tolerance in differences of set.
        maxiter : scalar(int), optional(default=500)
            Maximum number of iterations.
        check_pure_nash : bool, optional(default=True)
            Whether to perform a check about whether a pure Nash equilibrium
            exists.
        verbose : bool, optional(default=False)
            Whether to display updates about iterations and distance.
        nskipprint : scalar(int), optional(default=50)
            Number of iterations between printing information (assuming
            verbose=true).
        linprog_method : ,optional(default='simplex')
            The name of the scipy linprog method used to solve linear
            programming problems.

        Returns
        -------
        vertices : ndarray(float, ndim=2)
                   Vertices of the outer approximation of the value set.

        """
        sg, delta = self.sg, self.delta
        p1, p2 = sg.players
        po_1, po_2 = p1.payoff_array, p2.payoff_array
        p1_minpo, p1_maxpo = np.min(po_1.flatten()), np.max(po_1.flatten())
        p2_minpo, p2_maxpo = np.min(po_2.flatten()), np.max(po_2.flatten())

        try:
            next(pure_nash_brute_gen(sg))
        except StopIteration:
            raise ValueError("No pure action Nash equilibrium exists in" +
                             " stage game")

        # Get number of actions for each player and create action space
        nA1, nA2 = p1.num_actions, p2.num_actions
        nAS = nA1 * nA2
        AS = gridmake(np.array(range(nA1)), np.array(range(nA2)))

        # Create the unit circle, points, and hyperplane levels
        C, H, Z = initialize_sg_hpl(self, nH)
        Cnew = C.copy()

        # Create matrices for linear programming
        c, A, b = initialize_LP_matrices(self, H)

        # bounds on w are [-Inf, Inf] while bounds on slack are [0, Inf]
        lb = -np.inf
        ub = np.inf

        # Allocate space to store all solutions
        Cia = np.zeros(nAS)
        Wia = np.zeros([2, nAS])

        # Set iterative parameters and iterate until converged
        itr, dist = 0, 10.0
        while (itr < maxiter) and (dist > tol):
            # Compute the current worst values for each agent
            _w1 = worst_value_1(self, H, C)
            _w2 = worst_value_2(self, H, C)

            # Iterate over all subgradients
            for ih in range(nH):
                #
                # Subgradient specific instructions
                #
                h1, h2 = H[ih, :]

                # Update all set constraints - Copies elements 1:nH of C into b
                b[:nH] = C

                # Put the right objective into c (negative because want
                # maximize)
                c[0] = -h1
                c[1] = -h2

                for ia in range(nAS):
                    #
                    # Action specific instruction
                    #
                    a1, a2 = AS[ia, :]

                    # Update incentive constraints
                    b[nH] = (1-delta)*flow_u_1(self, a1, a2) - \
                            (1-delta)*best_dev_payoff_1(self, a2) - delta*_w1
                    b[nH+1] = (1-delta)*flow_u_2(self, a1, a2) - \
                              (1-delta)*best_dev_payoff_2(self, a1) - delta*_w2

                    lpout = linprog(c, A_ub=A, b_ub=b, bounds=(lb, ub),
                                    method=linprog_method)
                    if lpout.status == 0:
                        # Pull out optimal value and compute
                        w_sol = lpout.x
                        value = (1-delta)*flow_u(self, a1, a2) + delta*w_sol

                        # Save hyperplane level and continuation promises
                        Cia[ia] = h1*value[0] + h2*value[1]
                        Wia[:, ia] = value
                    else:
                        Cia[ia] = -np.inf

                # Action which pushes furthest in direction h_i
                astar = np.argmax(Cia)
                a1star, a2star = AS[astar, :]

                # Get hyperplane level and continuation value
                Cstar = Cia[astar]
                Wstar = Wia[:, astar]
                if Cstar > -1e10:
                    Cnew[ih] = Cstar
                else:
                    raise Error("Failed to find feasible action/continuation" +
                                " pair")

                # Update the points
                Z[:, ih] = \
                    (1-delta)*flow_u(self, a1star, a2star) + \
                    delta*np.array([Wstar[0], Wstar[1]])

            # Update distance and iteration counter
            dist = np.max(abs(C - Cnew))
            itr += 1

            if verbose and (nskipprint % itr == 0):
                println("$iter\t$dist\t($_w1, $_w2)")

            if itr >= maxiter:
                warn("Maximum Iteration Reached")

            # Update hyperplane levels
            C[:] = Cnew

        # Given the H-representation `(H, C)` of the computed polytope of
        # equilibrium payoff profiles, we obtain its V-representation
        # `vertices` using scipy
        p = HalfspaceIntersection(np.column_stack((H, -C)), np.mean(Z, axis=1))
        vertices = p.intersections

        # Reduce the number of vertices by rounding points to the tolerance
        tol_int = int(round(abs(np.log10(tol))) - 1)

        # Find vertices that are unique within tolerance level
        vertices = np.vstack({tuple(row) for row in
                              np.round(vertices, tol_int)})

        return vertices


def unitcircle(npts):
    """
    Places `npts` equally spaced points along the 2 dimensional circle and
    returns the points with x coordinates in first column and y coordinates
     in second column.

    Parameters
    ----------
    npts : scalar(float)
           Number of points.

    Returns
    -------
    pts : ndarray(float, dim=2)
          The coordinates of equally spaced points.

    """
    degrees = np.linspace(0, 2*np.pi, npts+1)

    pts = np.zeros((npts, 2))
    for i in range(npts):
        x = degrees[i]
        pts[i, 0] = np.cos(x)
        pts[i, 1] = np.sin(x)
    return pts


def initialize_hpl(nH, o, r):
    """
    Initializes subgradients, extreme points and hyperplane levels for the
    approximation of the convex value set of a 2 player repeated game.

    Parameters
    ----------
    nH : scalar(int)
        Number of subgradients used for the approximation.
    o : ndarray(float, ndim=2)
        Origin for the approximation.
    r: scalar(float)
       Radius for the approximation.

    Returns
    -------
    C : ndarray(float, ndim=1)
        The array containing the hyperplane levels.
    H : ndarray(float, ndim=2)
        The array containing the subgradients.
    Z : ndarray(float, ndim=2)
        The array containing the extreme points of the value set.

    """
    # Create unit circle
    H = unitcircle(nH)
    HT = np.transpose(H)

    # Choose origin and radius for big approximation
    Z = np.zeros([2, nH], dtype=float)
    for i in range(nH):
        # We know that players can ever get worse than their
        # lowest punishment, so ignore anything below that
        Z[0, i] = o[0] + r*HT[0, i]
        Z[1, i] = o[1] + r*HT[1, i]

    # Corresponding hyperplane levels
    C = np.squeeze(sum(np.multiply(HT, Z)))

    return C, H, Z


def initialize_sg_hpl(rpd, nH):
    """
    Initializes subgradients, extreme points and hyperplane levels for the
    approximation of the convex value set of a 2 player repeated game by
    choosing an appropriate origin and radius.

    Parameters
    ----------
    rpd : RepeatedGame
        Two player repeated game.
    nH : scalar(int)
        Number of subgradients used for the approximation.

    Returns
    -------
    C : ndarray(float, ndim=1)
        The array containing the hyperplane levels.
    H : ndarray(float, ndim=2)
        The array containing the subgradients.
    Z : ndarray(float, ndim=2)
        The array containing the extreme points of the value set.

    """
    # Choose the origin to be mean of max and min payoffs
    po_1 = rpd.sg.players[0].payoff_array.flatten()
    po_2 = rpd.sg.players[1].payoff_array.flatten()

    p1_min, p1_max = np.min(po_1), np.max(po_1)
    p2_min, p2_max = np.min(po_2), np.max(po_2)

    o = np.array([(p1_min + p1_max)/2.0, (p2_min + p2_max)/2.0])
    r1 = np.max(np.array((p1_max - o[0])**2, (o[0] - p1_min)**2))
    r2 = np.max(np.array((p2_max - o[1])**2, (o[1] - p2_min)**2))
    r = np.sqrt(r1 + r2)

    return initialize_hpl(nH, o, r)


def initialize_LP_matrices(rpd, H):
    """
    Initializes matrices for linear programming problems.

    Parameters
    ----------
    rpd : RepeatedGame
        Two player repeated game.
    H : ndarray(float, ndim=2)
        Subgradients used to approximate value set.

    Returns
    -------
    c : ndarray(float, ndim=1)
        Vector used to determine which subgradient is being used.
    A : ndarray(float, ndim=2)
        Matrix with nH set constraints and to be filled with 2 additional
        incentive compatibility constraints.
    b : ndarray(float, ndim=1)
        Vector to be filled with the values for the constraints.

    """
    # Total number of subgradients
    nH = H.shape[0]

    # Create the c vector (objective)
    c = np.zeros(2)

    # Create the A matrix (constraints)
    A = np.concatenate((H, [[-rpd.delta, 0.]], [[0., -rpd.delta]]))

    # Create the b vector (constraints)
    b = np.zeros(nH + 2)

    return c, A, b


# Flow utility in terms of the players actions
def flow_u_1(rpd, a1, a2): return rpd.sg.players[0].payoff_array[a1, a2]


def flow_u_2(rpd, a1, a2): return rpd.sg.players[1].payoff_array[a2, a1]


def flow_u(rpd, a1, a2):
    return np.array([flow_u_1(rpd, a1, a2), flow_u_2(rpd, a1, a2)])


# Computes each players best deviation given an opponent's action
def best_dev_i(rpd, i, aj):
    return np.argmax(rpd.sg.players[i].payoff_array[:, aj])


def best_dev_1(rpd, a2): return best_dev_i(rpd, 0, a2)


def best_dev_2(rpd, a1): return best_dev_i(rpd, 1, a1)


# Computes the payoff of the best deviation
def best_dev_payoff_i(rpd, i, aj):
    return max(rpd.sg.players[i].payoff_array[:, aj])


def best_dev_payoff_1(rpd, a2):
    return max(rpd.sg.players[0].payoff_array[:, a2])


def best_dev_payoff_2(rpd, a1):
    return max(rpd.sg.players[1].payoff_array[:, a1])


def worst_value_i(rpd, H, C, i, linprog_method='simplex'):
    """
    Returns the worst possible payoff for player i.

    Parameters
    ----------
    rpd : RepeatedGame
          Two player repeated game instance.
    H : ndarray(float, ndim=2)
        Subgradients used to approximate value set.
    C : ndarray(float, ndim=1)
        Hyperplane levels used to approximate the value set.
    i : scalar(int)
        The player of interest.

    Returns
    -------
    out : scalar(float)
          Worst possible payoff of player i.

    """
    # Objective depends on which player we are minimizing
    c = np.zeros(2)
    c[i] = 1.0

    # Lower and upper bounds for w
    lb = -np.inf
    ub = np.inf

    lpout = linprog(c, A_ub=H, b_ub=C, bounds=(lb, ub), method=linprog_method)
    if lpout.status == 0:
        out = lpout.x[i]
    else:
        out = np.min(rpd.sg.players[i].payoff_array)

    return out


def worst_value_1(rpd, H, C, linprog_method='simplex'):
    return worst_value_i(rpd, H, C, 0, linprog_method)


def worst_value_2(rpd, H, C, linprog_method='simplex'):
    return worst_value_i(rpd, H, C, 1, linprog_method)


def worst_values(rpd, H, C, linprog_method='simplex'):
    return (worst_value_1(rpd, H, C, linprog_method),
            worst_value_2(rpd, H, C, linprog_method))
