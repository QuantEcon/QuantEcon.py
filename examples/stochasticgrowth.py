"""
Neoclassical growth model with constant savings rate, where the dynamics are
given by

    k_{t+1} = s A_{t+1} f(k_t) + (1 - delta) k_t

Marginal densities are computed using the look-ahead estimator.  Thus, the
estimate of the density psi_t of k_t is

    (1/n) sum_{i=0}^n p(k_{t-1}^i, y)

This is a density in y.  
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm, beta
from quantecon import LAE

# == Define parameters == #
s = 0.2
delta = 0.1
a_sigma = 0.4       # A = exp(B) where B ~ N(0, a_sigma)
alpha = 0.4         # We set f(k) = k**alpha
psi_0 = beta(5, 5, scale=0.5)  # Initial distribution
phi = lognorm(a_sigma) 

def p(x, y):
    """
    Stochastic kernel for the growth model with Cobb-Douglas production.
    Both x and y must be strictly positive.
    """
    d = s * x**alpha
    return phi.pdf((y - (1 - delta) * x) / d) / d

n = 10000    # Number of observations at each date t
T = 30       # Compute density of k_t at 1,...,T+1

# == Generate matrix s.t. t-th column is n observations of k_t == #
k = np.empty((n, T))
A = phi.rvs((n, T))
k[:, 0] = psi_0.rvs(n)  # Draw first column from initial distribution
for t in range(T-1):
    k[:, t+1] = s * A[:,t] * k[:, t]**alpha + (1 - delta) * k[:, t]

# == Generate T instances of LAE using this data, one for each date t == #
laes = [LAE(p, k[:, t]) for t in range(T)]  

# == Plot == #
fig, ax = plt.subplots()
ygrid = np.linspace(0.01, 4.0, 200)
greys = [str(g) for g in np.linspace(0.0, 0.8, T)]
greys.reverse()
for psi, g in zip(laes, greys):
    ax.plot(ygrid, psi(ygrid), color=g, lw=2, alpha=0.6)
ax.set_xlabel('capital')
title = r'Density of $k_1$ (lighter) to $k_T$ (darker) for $T={}$'
ax.set_title(title.format(T))
plt.show()
