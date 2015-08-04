"""
Created on Mon Dec 16 19:12:17 2013
@author: dgevans
Edited by: Chase Coleman, John Stachurski

This file corresponds to the Ramsey model from the QE lecture on
history dependent policies:

    http://quant-econ.net/py/hist_dep_policies.html

In the following, ``uhat`` and ``tauhat`` are what the planner would choose if
he could reset at time t, ``uhatdif`` and ``tauhatdif`` are the difference
between those and what the planner is constrained to choose.  The variable
``mu`` is the Lagrange multiplier associated with the constraint at time t.

For more complete description of inputs and outputs see the website.

"""

import numpy as np
from quantecon import LQ
from quantecon.matrix_eqn import solve_discrete_lyapunov
from scipy.optimize import root


def computeG(A0, A1, d, Q0, tau0, beta, mu):
    """
    Compute government income given mu and return tax revenues and
    policy matrixes for the planner.

    Parameters
    ----------
    A0 : float
        A constant parameter for the inverse demand function
    A1 : float
        A constant parameter for the inverse demand function
    d : float
        A constant parameter for quadratic adjustment cost of production
    Q0 : float
        An initial condition for production
    tau0 : float
        An initial condition for taxes
    beta : float
        A constant parameter for discounting
    mu : float
        Lagrange multiplier

    Returns
    -------
    T0 : array(float)
        Present discounted value of government spending
    A : array(float)
        One of the transition matrices for the states
    B : array(float)
        Another transition matrix for the states
    F : array(float)
        Policy rule matrix
    P : array(float)
        Value function matrix
    """
    # Create Matrices for solving Ramsey problem
    R = np.array([[0, -A0/2, 0, 0],
                 [-A0/2, A1/2, -mu/2, 0],
                 [0, -mu/2, 0, 0],
                 [0, 0, 0, d/2]])

    A = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 1],
                 [0, 0, 0, 0],
                 [-A0/d, A1/d, 0, A1/d+1/beta]])

    B = np.array([0, 0, 1, 1/d]).reshape(-1, 1)

    Q = 0

    # Use LQ to solve the Ramsey Problem.
    lq = LQ(Q, -R, A, B, beta=beta)
    P, F, d = lq.stationary_values()

    # Need y_0 to compute government tax revenue.
    P21 = P[3, :3]
    P22 = P[3, 3]
    z0 = np.array([1, Q0, tau0]).reshape(-1, 1)
    u0 = -P22**(-1) * P21.dot(z0)
    y0 = np.vstack([z0, u0])

    # Define A_F and S matricies
    AF = A - B.dot(F)
    S = np.array([0, 1, 0, 0]).reshape(-1, 1).dot(np.array([[0, 0, 1, 0]]))

    # Solves equation (25)
    temp = beta * AF.T.dot(S).dot(AF)
    Omega = solve_discrete_lyapunov(np.sqrt(beta) * AF.T, temp)
    T0 = y0.T.dot(Omega).dot(y0)

    return T0, A, B, F, P


# == Primitives == #
T    = 20
A0   = 100.0
A1   = 0.05
d    = 0.20
beta = 0.95

# == Initial conditions == #
mu0  = 0.0025
Q0   = 1000.0
tau0 = 0.0


def gg(mu):
    """
    Computes the tax revenues for the government given Lagrangian
    multiplier mu.
    """
    return computeG(A0, A1, d, Q0, tau0, beta, mu)

# == Solve the Ramsey problem and associated government revenue == #
G0, A, B, F, P = gg(mu0)

# == Compute the optimal u0 == #
P21 = P[3, :3]
P22 = P[3, 3]
z0 = np.array([1, Q0, tau0]).reshape(-1, 1)
u0 = -P22**(-1) * P21.dot(z0)


# == Initialize vectors == #
y = np.zeros((4, T))
uhat       = np.zeros(T)
uhatdif    = np.zeros(T)
tauhat     = np.zeros(T)
tauhatdif  = np.zeros(T-1)
mu         = np.zeros(T)
G          = np.zeros(T)
GPay       = np.zeros(T)

# == Initial conditions == #
G[0] = G0
mu[0] = mu0
uhatdif[0] = 0
uhat[0] = u0
y[:, 0] = np.vstack([z0, u0]).flatten()

for t in range(1, T):
    # Iterate government policy
    y[:, t] = (A-B.dot(F)).dot(y[:, t-1])

    # update G
    G[t] = (G[t-1] - beta*y[1, t]*y[2, t])/beta
    GPay[t] = beta*y[1, t]*y[2, t]

    # Compute the mu if the government were able to reset its plan
    # ff is the tax revenues the government would receive if they reset the
    # plan with Lagrange multiplier mu minus current G

    ff = lambda mu: (gg(mu)[0]-G[t]).flatten()

    # find ff = 0
    mu[t] = root(ff, mu[t-1]).x
    temp, Atemp, Btemp, Ftemp, Ptemp = gg(mu[t])

    # Compute alternative decisions
    P21temp = Ptemp[3, :3]
    P22temp = P[3, 3]
    uhat[t] = -P22temp**(-1)*P21temp.dot(y[:3, t])

    yhat = (Atemp-Btemp.dot(Ftemp)).dot(np.hstack([y[0:3, t-1], uhat[t-1]]))
    tauhat[t] = yhat[3]
    tauhatdif[t-1] = tauhat[t]-y[3, t]
    uhatdif[t] = uhat[t]-y[3, t]


if __name__ == '__main__':
    print("1 Q tau u")
    print(y)
    print("-F")
    print(-F)
