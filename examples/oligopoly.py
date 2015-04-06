"""
Filename: oligopoly.py
Authors: Chase Coleman, Tom Sargent, Balint Szoke
This is an example for the lecture dyn_stack.rst from the QuantEcon
series of lectures by Tom Sargent and John Stachurski.
We deal with a large monopolistic firm who faces costs:
C_t = e Q_t + .5 g Q_t^2 + .5 c (Q_{t+1} - Q_t)^2
where the fringe firms face:
sigma_t = d q_t + .5 h q_t^2 + .5 c (q_{t+1} - q_t)^2
Additionally, there is a linear inverse demand curve of the form:
p_t = A_0 - A_1 (Q_t + \bar{q_t}) + \eta_t,
where:
.. math
    \eta_{t+1} = \rho \eta_t + C_{\varepsilon} \varepsilon_{t+1};
    \varepsilon_{t+1} \sim N(0, 1)
For more details, see the lecture.
"""
import numpy as np
import scipy.linalg as la
from quantecon import LQ
from quantecon.matrix_eqn import solve_discrete_lyapunov
from scipy.optimize import root


def setup_matrices(params):
    """
    This function sets up the A, B, R, Q for the oligopoly problem
    described in the lecture.
    Parameters
    ----------
    params : Array(Float, ndim=1)
        Contains the parameters that describe the problem in the order
        [a0, a1, rho, c_eps, c, d, e, g, h, beta]
    Returns
    -------
    (A, B, Q, R) : Array(Float, ndim=2)
        These matrices describe the oligopoly problem.
    """

    # Left hand side of (37)
    Alhs = np.eye(5)
    Alhs[4, :] = np.array([a0-d, 1., -a1, -a1-h, c])
    Alhsinv = la.inv(Alhs)

    # Right hand side of (37)
    Brhs = np.array([[0., 0., 1., 0., 0.]]).T
    Arhs = np.eye(5)
    Arhs[1, 1] = rho
    Arhs[3, 4] = 1.
    Arhs[4, 4] = c / beta

    # R from equation (40)
    R = np.array([[0., 0., (a0-e)/2., 0., 0.],
                  [0., 0., 1./2., 0., 0.],
                  [(a0-e)/2., 1./2, -a1 - .5*g, -a1/2, 0.],
                  [0., 0., -a1/2, 0., 0.],
                  [0., 0., 0., 0., 0.]])

    Rf = np.array([[0., 0., 0., 0., 0., (a0-d)/2.],
                  [0., 0., 0., 0., 0., 1./2.],
                  [0., 0., 0., 0., 0., -a1/2.],
                  [0., 0., 0., 0., 0., -a1/2.],
                  [0., 0., 0., 0., 0., 0.],
                  [(a0-d)/2., 1./2., -a1/2., -a1/2., 0., -h/2.]])

    Q = np.array([[c/2]])

    A = Alhsinv.dot(Arhs)
    B = Alhsinv.dot(Brhs)

    return A, B, Q, R, Rf


def find_PFd(A, B, Q, R, Rf, beta=.95):
    """
    Taking the parameters A, B, Q, R as found in the `setup_matrices`,
    we find the value function of the optimal linear regulator problem.
    This is steps 2 and 3 in the lecture notes.
    Parameters
    ----------
    (A, B, Q, R) : Array(Float, ndim=2)
        The matrices that describe the oligopoly problem
    Returns
    -------
    (P, F, d) : Array(Float, ndim=2)
        The matrix that describes the value function of the optimal
        linear regulator problem.
    """

    lq = LQ(Q, -R, A, B, beta=beta)
    P, F, d = lq.stationary_values()

    Af = np.vstack((np.hstack([A-np.dot(B,F), np.array([[0., 0., 0., 0., 0.]]).T]),np.array([[0., 0., 0., 0., 0., 1.]])))
    Bf = np.array([[0., 0., 0., 0., 0., 1.]]).T

    lqf = LQ(Q, -Rf, Af, Bf, beta=beta)
    Pf, Ff, df = lqf.stationary_values()

    return P, F, d, Pf, Ff, df


def solve_for_opt_policy(params, eta0=0., Q0=0., q0=0.):
    """
    Taking the parameters as given, solve for the optimal decision rules
    for the firm.
    Parameters
    ----------
    params : Array(Float, ndim=1)
        This holds all of the model parameters in an array
    Returns
    -------
    out :
    """
    # Step 1/2: Formulate/Solve the optimal linear regulator
    (A, B, Q, R, Rf) = setup_matrices(params)
    (P, F, d, Pf, Ff, df) = find_PFd(A, B, Q, R, Rf, beta=beta)

    # Step 3: Convert implementation into state variables (Find coeffs)
    P22 = P[-1, -1]
    P21 = P[-1, :-1]
    P22inv = P22**(-1)

    # Step 4: Find optimal x_0 and \mu_{x, 0}
    z0 = np.array([1., eta0, Q0, q0])
    x0 = -P22inv*np.dot(P21, z0)
    D0 = -np.dot(P22inv, P21)

    # Return -F and -Ff because we use u_t = -F y_t
    return P, -F, D0, Pf, -Ff


# Parameter values
a0 = 100.
a1 = 1.
rho = .8
c_eps = .2
c = 1.
d = 20.
e = 20.
g = .2
h = .2
beta = .95
params = np.array([a0, a1, rho, c_eps, c, d, e, g, h, beta])


P, F, D0, Pf,Ff  = solve_for_opt_policy(params)


# Checking time-inconsistency:
A, B, Q, R, Rf = setup_matrices(params)
# arbitrary initial z_0
y0 = np.array([[1, 1, 1, 1]]).T
# optimal x_0 = i_0
i0 = np.dot(D0,y0)
# iterate one period using the closed-loop system
y1 = np.dot( A + np.dot(B,F) , np.vstack([y0, i0]) )
# the last element of y_1 is x_1 = i_1
i1_0 = y1[-1,0]

# compare this to the case when the leader solves a Stackelberg problem
# in period 1. if in period 1 the leader could choose i1 given
# (1, v_1, Q_1, \bar{q}_1)
i1_1 = np.dot(D0, y1[0:-1,0])


print("P = {}".format(P))
print("-F = {}".format(F))
print("D0 = {}".format(D0))
print("Pf = {}".format(Pf))
print("-Ff = {}".format(Ff))
print("i1_0 = {}".format(i1_0))
print("i1_1 = {}".format(i1_1))
