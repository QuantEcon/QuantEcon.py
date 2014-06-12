import numpy as np
from scipy.stats import lognorm, beta
import matplotlib.pyplot as plt
from quantecon.lae import lae

# == Define parameters == #
s = 0.2
delta = 0.1
a_sigma = 0.4       # A = exp(B) where B ~ N(0, a_sigma)
alpha = 0.4         # f(k) = k^{\alpha}

phi = lognorm(a_sigma) 

def p(x, y):
    "Stochastic kernel, vectorized in x.  Both x and y must be positive."
    d = s * x**alpha
    return phi.pdf((y - (1 - delta) * x) / d) / d

n = 1000     # Number of observations at each date t
T = 40       # Compute density of k_t at 1,...,T

fig, axes = plt.subplots(2, 2)
axes = axes.flatten()
xmax = 6.5

for i in range(4):
    ax = axes[i] 
    ax.set_xlim(0, xmax)
    psi_0 = beta(5, 5, scale=0.5, loc=i*2)  # Initial distribution

    # == Generate matrix s.t. t-th column is n observations of k_t == #
    k = np.empty((n, T))
    A = phi.rvs((n, T))
    k[:, 0] = psi_0.rvs(n)
    for t in range(T-1):
        k[:, t+1] = s * A[:,t] * k[:, t]**alpha + (1 - delta) * k[:, t]

    # == Generate T instances of lae using this data, one for each t == #
    laes = [lae(p, k[:, t]) for t in range(T)]

    ygrid = np.linspace(0.01, xmax, 150)
    greys = [str(g) for g in np.linspace(0.0, 0.8, T)]
    greys.reverse()
    for psi, g in zip(laes, greys):
        ax.plot(ygrid, psi(ygrid), color=g, lw=2, alpha=0.6)
    #ax.set_xlabel('capital')
    #title = r'Density of $k_1$ (lighter) to $k_T$ (darker) for $T={}$'
    #ax.set_title(title.format(T))

plt.show()

