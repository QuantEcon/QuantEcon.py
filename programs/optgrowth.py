"""
Origin: QE by John Stachurski and Thomas J. Sargent
Filename: optgrowth.py
Authors: John Stachurski and Thomas Sargent
LastModified: 11/08/2013

Solving the optimal growth problem via value function iteration.

"""

from __future__ import division  # Omit for Python 3.x
import numpy as np
from scipy.optimize import fminbound
from scipy import interp

class growthModel:
    """
    This class is just a "struct" to hold the collection of primitives
    defining the growth model.  The default values are 

        f(k) = k**alpha, i.e, Cobb-douglas production function
        u(c) = ln(c), i.e, log utility

    See the __init__ function for details
    """
    def __init__(self, f=lambda k: k**0.65, beta=0.95, u=np.log, 
            grid_max=2, grid_size=150):
        """
        Parameters:

            * f is the production function and u is the utility function 
            * beta is the discount factor, a scalar in (0, 1)
            * grid_max and grid_size describe the grid 

        """
        self.u, self.f, self.beta = u, f, beta
        self.grid = np.linspace(1e-6, grid_max, grid_size)


def bellman_operator(gm, w):
    """
    The approximate Bellman operator, which computes and returns the updated
    value function Tw on the grid poitns.

    Parameters:

        * gm is an instance of the growthModel class
        * w is a flat NumPy array with len(w) = len(grid)

    The vector w represents the value of the input function on the grid
    points.

    """
    # === Apply linear interpolation to w === #
    Aw = lambda x: interp(x, gm.grid, w)  

    # === set Tw[i] equal to max_c { u(c) + beta w(f(k_i) - c)} === #
    Tw = np.empty(len(w))
    for i, k in enumerate(gm.grid):
        objective = lambda c:  - gm.u(c) - gm.beta * Aw(gm.f(k) - c)
        c_star = fminbound(objective, 1e-6, gm.f(k))
        Tw[i] = - objective(c_star)

    return Tw


def compute_greedy(gm, w):
    """
    Compute the w-greedy policy on the grid points.  Parameters:

        * gm is an instance of the growthModel class
        * w is a flat NumPy array with len(w) = len(grid)

    """
    # === Apply linear interpolation to w === #
    Aw = lambda x: interp(x, gm.grid, w)  

    # === set sigma[i] equal to argmax_c { u(c) + beta w(f(k_i) - c)} === #
    sigma = np.empty(len(w))
    for i, k in enumerate(gm.grid):
        objective = lambda c:  - gm.u(c) - gm.beta * Aw(gm.f(k) - c)
        sigma[i] = fminbound(objective, 1e-6, gm.f(k))

    return sigma

