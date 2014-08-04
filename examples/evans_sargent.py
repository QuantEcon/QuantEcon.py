"""
Created on Mon Dec 16 19:12:17 2013
@author: dgevans
Edited by: Chase Coleman, John Stachurski

This file corresponds to the Ramsey model from the QE lecture on
history dependent policies:

    http://quant-econ.net/hist_dep_policies.html

For more complete description of inputs and outputs see the website.

"""

from numpy import array, vstack, sqrt, zeros, arange, hstack
from quantecon import LQ
from scipy.linalg import solve_discrete_lyapunov
from scipy.optimize import root
import matplotlib.pyplot as plt

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
    R = array([[0, -A0/2, 0, 0],
               [-A0/2, A1/2, -mu/2, 0],
               [0, -mu/2, 0, 0],
               [0, 0, 0, d/2]])

    A = array([[1, 0, 0, 0],
               [0, 1, 0, 1],
               [0, 0, 0, 0],
               [-A0/d, A1/d, 0, A1/d+1/beta]])

    B = array([0, 0, 1, 1/d]).reshape(-1, 1)

    Q = 0

    # Use LQ to solve the Ramsey Problem.
    lq = LQ(Q, -R, A, B, beta=beta)
    P, F, d = lq.stationary_values()

    # Need y_0 to compute government tax revenue.
    P21 = P[3, :3]
    P22 = P[3, 3]
    z0 = array([1, Q0, tau0]).reshape(-1, 1)
    u0 = -P22**(-1)*P21.dot(z0)
    y0 = vstack([z0, u0])

    # Define A_F and S matricies
    AF = A-B.dot(F)
    S = array([0, 1, 0, 0]).reshape(-1, 1).dot(array([[0, 0, 1, 0]]))

    # Solves equation (25)
    Omega = solve_discrete_lyapunov(sqrt(beta)*AF.T, beta*AF.T.dot(S).dot(AF))
    T0 = y0.T.dot(Omega).dot(y0)

    return T0, A, B, F, P


# == Parameters == #
T    = 20
A0   = 100.0
A1   = 0.05
d    = 0.20
mu0  = 0.0025
beta = 0.95
Q0   = 1000.0
tau0 = 0.0

def gg(mu):
    """
    Computes the tax revenues for the government given Lagrangian
    multiplier mu.
    """
    return computeG(A0, A1, d, Q0, tau0, beta, mu)

# == Solve the Ramsey problem and associated government revenue for mu0 == #
G0, A, B, F, P = gg(mu0)

# == Compute the optimal u0 == #
P21 = P[3, :3]
P22 = P[3, 3]
z0 = array([1, Q0, tau0]).reshape(-1, 1)
u0 = -P22**(-1)*P21.dot(z0)

# == Set up variables to compute time series (t=1 refers to time 0) == #
y = zeros((4, T))

# uhat and tauhat are what the planner would choose if he could reset at
# time t, uhatdif and tauhatdif are the difference between those and
# what the planner is constrained to choose.
uhat = zeros(T)
uhatdif = zeros(T)
tauhat = zeros(T)
tauhatdif = zeros(T-1)
uhatdif[0] = 0
uhat[0] = u0
y[:, 0] = vstack([z0, u0]).flatten()

# mu is the Lagrange multiplier associated with the constraint at time t.
mu = zeros(T)
mu[0] = mu0
G = zeros(T)
G[0] = G0
GPay = zeros(T)
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

    # tauhat(t) = -F*[y(1:3,t-1);uhat(t-1)];
    yhat = (Atemp-Btemp.dot(Ftemp)).dot(hstack([y[0:3, t-1], uhat[t-1]]))
    tauhat[t] = yhat[3]
    tauhatdif[t-1] = tauhat[t]-y[3, t]
    uhatdif[t] = uhat[t]-y[3, t]

print("1 Q tau u")
print(y)
print("-F")
print(-F)

# Plots.  Note tt is used to make the plot time index correct.
plt.figure(1)
plt.plot(y[2, :])
plt.title("taxes")
tt = arange(T)

plt.figure(2)

plt.subplot(2, 2, 1)
plt.plot(tt, y[1, :])
plt.ylabel("Q")
plt.xlabel("t")

plt.subplot(2, 2, 2)
plt.plot(tt, y[2, :])
plt.ylabel(r"$\tau$")
plt.xlabel("t")

plt.subplot(2, 2, 3)
plt.plot(tt, y[3, :])
plt.ylabel("u")
plt.xlabel("t")

plt.figure(3)
plt.plot(tt, y[1, :])

plt.figure(4)
lines = plt.plot(tt, uhatdif, "o")
plt.setp(lines, "ms", 7., "markerfacecolor", "b")
plt.ylabel(r"$\Delta u$")
plt.xlabel("t")
plt.axis([-0.5, 20, -3, .1])

plt.figure(5)
tt2 = arange(T-1)
lines = plt.plot(tt2, tauhatdif, "o")
plt.setp(lines, "MarkerSize", 7, "MarkerFaceColor", "b")
plt.ylabel(r"$\Delta\tau$")
plt.xlabel("t")
plt.axis([-0.5, 20, -0.1, 1.4])

plt.figure(6)
lines = plt.plot(tt, mu, "o")
plt.setp(lines, "MarkerSize", 7, "MarkerFaceColor", "b")
plt.ylabel(r"$\mu$")
plt.xlabel("t")
plt.axis([-0.5, 20, 2.34e-3, 2.52e-3])

plt.figure(7)
lines = plt.plot(tt, G, "o")
plt.setp(lines, "MarkerSize", 7, "MarkerFaceColor", "b")
plt.ylabel("G")
plt.xlabel("t")
plt.axis([-0.5, 20, 9100, 9800])
plt.figure(8)
lines = plt.plot(tt, GPay, "o")
plt.setp(lines, "MarkerSize", 7)
