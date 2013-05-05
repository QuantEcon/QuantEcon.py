"""
Origin: QEwP by John Stachurski and Thomas J. Sargent
Date: 3/2013
File: ifp.py

Functions for solving the income fluctuation problem. Iteration with either
the Coleman or Bellman operators from appropriate initial conditions leads to
convergence to the optimal consumption policy.  The income process is a finite
state Markov chain.  Note that the Coleman operator is the preferred method,
as it is almost always faster and more accurate.  The Bellman operator is
only provided for comparison.

"""

import numpy as np
from scipy.optimize import fminbound, brentq
from scipy import interp

class consumerProblem:
    """
    This class is just a "struct" to hold the collection of parameters
    defining the consumer problem.  
    """

    def __init__(self, 
            r=0.01, beta=0.96, 
            Pi=((0.6, 0.4), (0.05, 0.95)), 
            z_vals=(0.5, 1.0), 
            b=0, grid_max=16, grid_size=50,
            u=np.log, 
            d_u=lambda x: 1/x):
        """
        Parameters:

            * r and beta are scalars with r > 0 and (1 + r) * beta < 1
            * Pi is a 2D NumPy array --- the Markov matrix for {z_t}
            * z_vals is an array/list containing the state space of {z_t}
            * u is the utility function and d_u is the derivative
            * b is the borrowing constraint
            * grid_max and grid_size describe the grid used in the solution

        """
        self.u, self.d_u = u, d_u
        self.r, self.R = r, 1 + r
        self.beta, self.b = beta, b
        self.Pi, self.z_vals = np.array(Pi), z_vals
        self.asset_grid = np.linspace(-b, grid_max, grid_size)


def bellman_operator(m, V, return_policy=False):
    """
    The approximate Bellman operator, which computes and returns the updated
    value function TV (or the V-greedy policy c if return_policy == True).

    Parameters:

        * m is an instance of class consumerProblem
        * V is a NumPy array of dimension len(m.asset_grid) x len(m.z_vals)

    """
    R, Pi, beta, u, b = m.R, m.Pi, m.beta, m.u, m.b  # Simplify parameter names
    new_V = np.empty(V.shape)
    new_c = np.empty(V.shape)
    # Turn V into a function based on linear interpolation along the asset grid
    vf = lambda a, i_z: interp(a, m.asset_grid, V[:, i_z]) 
    z_index = range(len(m.z_vals))  

    for i_a, a in enumerate(m.asset_grid):
        for i_z, z in enumerate(m.z_vals):
            def obj(c):  # Define objective function to be *minimized*
                y = sum(vf(R * a + z - c, j) * Pi[i_z, j] for j in z_index)
                return - u(c) - beta * y
            c_star = fminbound(obj, np.min(m.z_vals), R * a + z + b)
            new_c[i_a, i_z], new_V[i_a, i_z] = c_star, -obj(c_star)

    if return_policy:
        return new_c
    else:
        return new_V

def coleman_operator(m, c):
    """
    The approximate Coleman operator.  Iteration with this operator
    corresponds to policy function iteration.  Computes and returns the
    updated consumption policy c.

    Parameters:

        * m is an instance of class consumerProblem
        * c is a NumPy array of dimension len(m.asset_grid) x len(m.z_vals)

    The array c is replaced with a function cf that implements linear
    interpolation over the points between the asset grid.
    """
    R, Pi, beta, d_u, b = m.R, m.Pi, m.beta, m.d_u, m.b
    gamma = R * beta
    new_c = np.empty(c.shape)
    # Turn c into a function based on linear interpolation along the asset grid
    cf = lambda a, i_z: interp(a, m.asset_grid, c[:, i_z])
    z_index = range(len(m.z_vals))

    for i_a, a in enumerate(m.asset_grid):
        for i_z, z in enumerate(m.z_vals):
            def h(t):
                y = sum(d_u(cf(R*a + z - t, j)) * Pi[i_z,j] for j in z_index)
                return d_u(t) - max(gamma * y, d_u(R*a + z + b))
            # Compute zero of the function h and record it
            new_c[i_a, i_z] = brentq(h, np.min(m.z_vals), R * a + z + b)

    return new_c

def initialize(m):
    """
    Creates a suitable initial condition V for value function iteration, and a
    suitable consumption policy c for policy function iteration.

        * m is an instance of class consumerProblem.
    """
    N = len(m.z_vals)         
    K = len(m.asset_grid)    
    V, c = np.empty((K, N)), np.empty((K, N))
    for i_a, a in enumerate(m.asset_grid):
        for i_z, z in enumerate(m.z_vals):
            c_max = m.R * a + z + m.b
            c[i_a, i_z] = c_max
            V[i_a, i_z] = m.u(c_max) / (1 - m.beta)
    return V, c


