"""
Origin: QE by John Stachurski and Thomas J. Sargent
Filename: lae.py
Authors: John Stachurski, Thomas J. Sargent
LastModified: 21/08/2013

Computes a sequence of marginal densities for a stochastic neoclassical growth
model with constant savings rate, where the dynamics are given by

    k_{t+1} = s A_t f(k_t) + (1 - delta) k_t

Marginal densities are computed using the look-ahead estimator.  Thus, the
estimate of the density psi_t of k_t is

    (1/n) sum_{i=0}^n p(k_{t-1}^i, y)

This is a density in y.  
"""

import numpy as np
from scipy.stats import lognorm, beta
import matplotlib.pyplot as plt

class lae:
    """
    An instance is a representation of a look ahead estimator associated with
    a given stochastic kernel p and a vector of observations X.  For example,

    >>> psi = lae(p, X)
    >>> y = np.linspace(0, 1, 100)
    >>> psi(y)  # Evaluate look ahead estimate at grid of points y
    """

    def __init__(self, p, X):
        """
        Parameters
        ==========
        p : function
            The stochastic kernel.  A function p(x, y) that is vectorized in
            both x and y

        X : array_like
            A vector containing observations
        """
        X = X.flatten()  # So we know what we're dealing with
        n = len(X)
        self.p, self.X = p, X.reshape((n, 1))


    def __call__(self, y):
        """
        Parameters
        ==========
        y : array_like
            A vector of points at which we wish to evaluate the look-ahead
            estimator

        Returns
        =======
        psi_vals : numpy.ndarray
            The values of the density estimate at the points in y

        """
        k = len(y)
        v = self.p(self.X, y.reshape((1, k)))
        psi_vals = np.mean(v, axis=0)    # Take mean along each row
        return psi_vals.flatten()



# == An Example: Stochastic growth with Cobb-Douglas production == #

if __name__ == '__main__':  # If run directly, not imported 

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

    # == Generate T instances of lae using this data, one for each date t == #
    laes = [lae(p, k[:, t]) for t in range(T)]  

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
    fig.show()
