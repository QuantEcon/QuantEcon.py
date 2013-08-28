"""
Origin: QE by John Stachurski and Thomas J. Sargent
Filename: kalman_test.py
Authors: John Stachurski and Thomas Sargent
LastModified: 11/08/2013

A test with multivariate parameters.
"""
import numpy as np
from numpy.random import multivariate_normal
import matplotlib.pyplot as plt
from kalman import Kalman

# === define prior density === #
Sigma = [[0.4, 0.3], [0.3, 0.45]]
Sigma = np.array(Sigma)
x_hat = np.array([0.2, -0.2])

# === define A, Q, G, R === #
G = np.eye(2)
R = 0.5 * Sigma
A = np.eye(2)
Q = np.zeros(2)

# === initialize Kalman filter === #
kn = Kalman(A, G, Q, R)
kn.set_state(x_hat, Sigma)

# === the true value of the state === #
theta = np.zeros(2) + 4.0

# === generate plot === #
T = 1000
z = np.empty(T)
for t in range(T):
    # Measure the error
    y = multivariate_normal(mean=theta, cov=R)
    kn.update(y)
    z[t] = np.sqrt(np.sum((theta - kn.current_x_hat)**2))

fig, ax = plt.subplots()
ax.plot(range(T), z) 
ax.fill_between(range(T), np.zeros(T), z, color="blue", alpha=0.2) 
fig.show()


