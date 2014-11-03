"""
Filename: optgrowth.py

Authors: John Stachurski and Thomas Sargent

Solving the optimal growth problem via value function iteration.

"""

from __future__ import division  # Omit for Python 3.x
import numpy as np
from scipy.optimize import fminbound
from scipy import interp

class GrowthModel(object):
    """

    This class defines the primitives representing the growth model.

    Parameters
    ----------
    f : function, optional(default=k**.65)
        The production function; they default is the Cobb-Douglas
        production function with power of .65
    beta : scalar(int), optional(default=.95)
        The utility discounting parameter
    u : function, optional(default=np.log)
        The utility function.  Default is log utility
    grid_max : scalar(int), optional(default=2)
        The maximum grid value
    grid_size : scalar(int), optional(default=150)
        The size of grid to use.

    Attributes
    ----------
    f : function
        The production function
    beta : scalar(int)
        The utility discounting parameter
    u : function
        The utility function.
    grid : array_like(float, ndim=1)
        The grid over savings.

    """
    def __init__(self, f=lambda k: k**0.65, beta=0.95, u=np.log,
            grid_max=2, grid_size=150):

        self.u, self.f, self.beta = u, f, beta
        self.grid = np.linspace(1e-6, grid_max, grid_size)


    def bellman_operator(self, w, compute_policy=False):
        """
        The approximate Bellman operator, which computes and returns the
        updated value function Tw on the grid points.

        Parameters
        ----------
        w : array_like(float, ndim=1)
            The value of the input function on different grid points
        compute_policy : Boolean, optional(default=False)
            Whether or not to compute policy function

        """
        # === Apply linear interpolation to w === #
        Aw = lambda x: interp(x, self.grid, w)

        if compute_policy:
            sigma = np.empty(len(w))

        # == set Tw[i] equal to max_c { u(c) + beta w(f(k_i) - c)} == #
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
        Compute the w-greedy policy on the grid points.

        Parameters
        ----------
        w : array_like(float, ndim=1)
            The value of the input function on different grid points

        """
        Tw, sigma = self.bellman_operator(w, compute_policy=True)
        return sigma
