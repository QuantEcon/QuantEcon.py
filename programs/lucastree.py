"""
Origin: QE by Thomas J. Sargent and John Stachurski 
Filename: lucastree.py
Authors: Thomas Sargent, John Stachurski
LastModified: Tue May 6 08:37:18 EST 2014

Solves the price function for the Lucas tree in a continuous state setting,
using piecewise linear approximation for the sequence of candidate price
functions.  The consumption endownment follows the log linear AR(1) process

    log y' = alpha log y + sigma epsilon

where y' is a next period y and epsilon is an iid standard normal shock.
Hence

    y' = y^alpha * xi   where xi = e^(sigma * epsilon)

The distribution phi of xi is

    phi = LN(0, sigma^2) where LN means lognormal

Example usage:

    tree = lucas_tree(gamma=2, beta=0.95, alpha=0.90, sigma=0.1)
    grid, price_vals = compute_price(tree)


"""

from __future__ import division  # Omit for Python 3.x
import numpy as np
from collections import namedtuple
from scipy import interp
from scipy.stats import lognorm
from scipy.integrate import fixed_quad

# == Use a namedtuple to store the parameters of the Lucas tree == #

lucas_tree = namedtuple('lucas_tree', 
        ['gamma',   # Risk aversion
         'beta',    # Discount factor
         'alpha',   # Correlation coefficient
         'sigma'])  # Shock volatility

# == A function to compute the price == #

def compute_price(lt, grid=None):
    """
    Compute the equilibrium price function associated Lucas tree lt

    Parameters
    ==========
    lt : namedtuple, lucas_tree
        A namedtuple containing the parameters of the Lucas tree

    grid : a NumPy array giving the grid points on which to return the
        function values.  Grid points should be nonnegative.

    """
    # == Simplify names, set up distribution phi == #
    gamma, beta, alpha, sigma = lt.gamma, lt.beta, lt.alpha, lt.sigma
    phi = lognorm(sigma)

    # == Set up a function for integrating w.r.t. phi == #

    int_min, int_max = np.exp(-4 * sigma), np.exp(4 * sigma)  
    def integrate(g):
        "Integrate over three standard deviations"
        integrand = lambda z: g(z) * phi.pdf(z)
        result, error = fixed_quad(integrand, int_min, int_max)
        return result

    # == If there's no grid, form an appropriate one == #

    if grid == None:
        grid_size = 100
        if abs(alpha) >= 1:
            # If nonstationary, put the grid on [0,10]
            grid_min, grid_max = 0, 10
        else:
            # Set the grid interval to contain most of the mass of the
            # stationary distribution of the consumption endowment
            ssd = sigma / np.sqrt(1 - alpha**2)
            grid_min, grid_max = np.exp(-4 * ssd), np.exp(4 * ssd)
        grid = np.linspace(grid_min, grid_max, grid_size)
    else:
        grid_min, grid_max, grid_size = min(grid), max(grid), len(grid)

    # == Compute the function h in the Lucas operator as a vector of == # 
    # == values on the grid == #

    h = np.empty(grid_size)
    # Recall that h(y) = beta * int u'(G(y,z)) G(y,z) phi(dz)
    for i, y in enumerate(grid):
        integrand = lambda z: (y**alpha * z)**(1 - gamma) # u'(G(y,z)) G(y,z)
        h[i] = beta * integrate(integrand)

    # == Set up the Lucas operator T == #

    def lucas_operator(f):
        """
        The approximate Lucas operator, which computes and returns the updated
        function Tf on the grid poitns.

        Parameters
        ==========

        f : flat NumPy array with len(f) = len(grid)
            A candidate function on R_+ represented as points on a grid 

        """
        Tf = np.empty(len(f))
        Af = lambda x: interp(x, grid, f)  # Piecewise linear interpolation

        for i, y in enumerate(grid):
            Tf[i] = h[i] + beta * integrate(lambda z: Af(y**alpha * z))

        return Tf

    # == Now compute the price by iteration == #

    error_tol, max_iter = 1e-3, 50
    error = error_tol + 1
    iterate = 0
    f = np.zeros(len(grid))  # Initial condition
    while iterate < max_iter and error > error_tol:
        new_f = lucas_operator(f)
        iterate += 1
        error = np.max(np.abs(new_f - f))
        print error
        f = new_f

    return grid, f * grid**gamma # p(y) = f(y) / u'(y) = f(y) * y^gamma



