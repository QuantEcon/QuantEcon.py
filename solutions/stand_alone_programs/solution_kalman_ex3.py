from __future__ import print_function  # Remove for Python 3.x
import numpy as np
from numpy.random import multivariate_normal
import matplotlib.pyplot as plt
from scipy.linalg import eigvals
from quantecon.kalman import Kalman

# === Define A, Q, G, R === #
G = np.eye(2)
R = 0.5 * np.eye(2)
A = [[0.5, 0.4], 
     [0.6, 0.3]]
Q = 0.3 * np.eye(2)

# === Define the prior density === #
Sigma = [[0.9, 0.3], 
         [0.3, 0.9]]
Sigma = np.array(Sigma)
x_hat = np.array([8, 8])

# === Initialize the Kalman filter === #
kn = Kalman(A, G, Q, R)
kn.set_state(x_hat, Sigma)

# === Set the true initial value of the state === #
x = np.zeros(2)

# == Print eigenvalues of A == #
print("Eigenvalues of A:")
print(eigvals(A))

# == Print stationary Sigma == #
S, K = kn.stationary_values()
print("Stationary prediction error variance:")
print(S)

# === Generate the plot === #
T = 50
e1 = np.empty(T)
e2 = np.empty(T)
for t in range(T):
    # == Generate signal and update prediction == #
    y = multivariate_normal(mean=np.dot(G, x), cov=R)
    kn.update(y)
    # == Update state and record error == #
    Ax = np.dot(A, x)
    x = multivariate_normal(mean=Ax, cov=Q)
    e1[t] = np.sum((x - kn.current_x_hat)**2)
    e2[t] = np.sum((x - Ax)**2)

fig, ax = plt.subplots()
ax.plot(range(T), e1, 'k-', lw=2, alpha=0.6, label='Kalman filter error') 
ax.plot(range(T), e2, 'g-', lw=2, alpha=0.6, label='conditional expectation error') 
ax.legend()
plt.show()

