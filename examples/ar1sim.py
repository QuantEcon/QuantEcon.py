"""
Origin: QE by Thomas J. Sargent and John Stachurski
Filename: ar1sim.py
"""

import numpy as np
from scipy.stats import norm

def proto1(a, b, sigma, T, num_reps, phi=norm.rvs):
    X = np.zeros((num_reps, T+1))
    for i in range(num_reps):
        W = phi(size=T+1)
        for t in range(1, T+1):
            X[i, t] = a * X[i,t-1] + b + W[t]
    return X


def proto2(a, b, sigma, T, num_reps, x0=None, phi=norm.rvs):
    """
    More efficient, eliminates one loop.
    """
    if not x0 == None:
        x0.shape = (num_reps, 1)
        X[:, 0] = x0
    W = phi(size=(num_reps, T+1))
    X = np.zeros((num_reps, T+1))
    for t in range(1, T+1):
        X[:, t] = a * X[:,t-1] + b + W[:, t]
    return X

def ols_estimates(X):
    num_reps, ts_length = X.shape
    estimates = np.empty(num_reps)
    for i in range(num_reps):
        X_row = X[i,:].flatten()
        x = X_row[:-1]  # All but last one
        y = X_row[1:]   # All but first one
        estimates[i] = np.dot(x, y) / np.dot(x, x)
    return estimates

def ope_estimates(X):
    num_reps, ts_length = X.shape
    estimates = np.empty(num_reps)
    for i in range(num_reps):
        x = X[i,:].flatten()
        s2 = x.var()
        estimates[i] = np.sqrt(1 - 1 / s2)
    return estimates

theta = 0.8
num_reps = 100000
n = 1000
X_obs = proto2(theta, 0, 1, n, num_reps)

if 0:
    theta_hats = ols_estimates(X_obs)
    r = np.sqrt(n) * (theta_hats - theta)
    print("OLS Expected: {}".format(1 - theta**2))
    print("OLS Realized: {}".format(r.var()))

if 0:
    theta_hats = ope_estimates(X_obs)
    r = np.sqrt(n) * (theta_hats - theta)
    e = (1 - theta**2) * (1 + (1 - theta**2) / (2 * theta**2))
    print("OPE Expected: {}".format(e))
    print("OPE Realized: {}".format(r.var()))

s2_hats = X_obs.var(axis=1)
r = np.sqrt(n) * (s2_hats - 1 / (1 - theta**2))
e = 2 * (1 + theta**2) / (1 - theta**2)**3
print("Expected: {}".format(e))
print("Realized: {}".format(r.var()))


