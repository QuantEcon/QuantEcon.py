"""
Filename: optgrowth.py
Authors: John Stachurski and Thomas Sargent

Solving the optimal growth problem via value function iteration.

"""

from __future__ import division  # Omit for Python 3.x
import numpy as np
from scipy.optimize import fminbound
from scipy import interp

class GrowthModel:
    """
    This class defines the primitives representing the growth model.  The
    default values are 

        f(k) = k**alpha, i.e, Cobb-Douglas production function
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


    def bellman_operator(self, w, compute_policy=False):
        """
        The approximate Bellman operator, which computes and returns the 
        updated value function Tw on the grid points.

        Parameters
        ==========
            w : a flat NumPy array with len(w) = len(grid)

        The vector w represents the value of the input function on the grid
        points.

        """
        # === Apply linear interpolation to w === #
        Aw = lambda x: interp(x, self.grid, w)  

        if compute_policy:
            sigma = np.empty(len(w))

        # === set Tw[i] equal to max_c { u(c) + beta w(f(k_i) - c)} === #
        Tw = np.empty(len(w))
        for i, k in enumerate(self.grid):
            objective = lambda c:  - self.u(c) - self.beta * Aw(self.f(k) - c)
            c_star = fminbound(objective, 1e-6, self.f(k))
            if compute_policy:
                # sigma[i] = argmax_c { u(c) + beta w(f(k_i) - c)} 
                sigma[i] = c_star
            Tw[i] = - objective(c_star)

        if compute_policy:
            return Tw, sigma
        else:
            return Tw


    def compute_greedy(self, w):
        """
        Compute the w-greedy policy on the grid points.  Parameters:

        Parameters
        ==========
            w : a flat NumPy array with len(w) = len(grid)

        """
        Tw, sigma = self.bellman_operator(w, compute_policy=True)
        return sigma

