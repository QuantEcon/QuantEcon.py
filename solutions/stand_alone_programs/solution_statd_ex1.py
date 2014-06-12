"""
Look ahead estimation of a TAR stationary density, where the TAR model is

    X' = theta |X| + sqrt(1 - theta^2) xi

and xi is standard normal.  Try running at n = 10, 100, 1000, 10000 to get an
idea of the speed of convergence.
"""
import numpy as np
from scipy.stats import norm, gaussian_kde
import matplotlib.pyplot as plt
from quantecon.lae import lae

phi = norm()
n = 500
theta = 0.8
# == Frequently used constants == #
d = np.sqrt(1 - theta**2) 
delta = theta / d

def psi_star(y):
    "True stationary density of the TAR Model"
    return 2 * norm.pdf(y) * norm.cdf(delta * y) 

def p(x, y):
        "Stochastic kernel for the TAR model."
        return phi.pdf((y - theta * np.abs(x)) / d) / d

Z = phi.rvs(n)
X = np.empty(n)
for t in range(n-1):
    X[t+1] = theta * np.abs(X[t]) + d * Z[t]
psi_est = lae(p, X)
k_est = gaussian_kde(X)

fig, ax = plt.subplots()
ys = np.linspace(-3, 3, 200)
ax.plot(ys, psi_star(ys), 'b-', lw=2, alpha=0.6, label='true')
ax.plot(ys, psi_est(ys), 'g-', lw=2, alpha=0.6, label='look ahead estimate')
ax.plot(ys, k_est(ys), 'k-', lw=2, alpha=0.6, label='kernel based estimate')
ax.legend(loc='upper left')
plt.show()
