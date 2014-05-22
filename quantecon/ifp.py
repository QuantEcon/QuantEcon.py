"""
Filename: ifp.py
Authors: Thomas Sargent, John Stachurski 

Functions for solving the income fluctuation problem. Iteration with either
the Coleman or Bellman operators from appropriate initial conditions leads to
convergence to the optimal consumption policy.  The income process is a finite
state Markov chain.  Note that the Coleman operator is the preferred method,
as it is almost always faster and more accurate.  The Bellman operator is only
provided for comparison.

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
            r=0.01, 
            beta=0.96, 
            Pi=((0.6, 0.4), (0.05, 0.95)), 
            z_vals=(0.5, 1.0), 
            b=0, 
            grid_max=16, 
            grid_size=50,
            u=np.log, 
            du=lambda x: 1/x):
        """
        Parameters:

            * r and beta are scalars with r > 0 and (1 + r) * beta < 1
            * Pi is a 2D NumPy array --- the Markov matrix for {z_t}
            * z_vals is an array/list containing the state space of {z_t}
            * u is the utility function and du is the derivative
            * b is the borrowing constraint
            * grid_max and grid_size describe the grid used in the solution

        """
        self.u, self.du = u, du
        self.r, self.R = r, 1 + r
        self.beta, self.b = beta, b
        self.Pi, self.z_vals = np.array(Pi), tuple(z_vals)
        self.asset_grid = np.linspace(-b, grid_max, grid_size)


def bellman_operator(cp, V, return_policy=False):
    """
    The approximate Bellman operator, which computes and returns the updated
    value function TV (or the V-greedy policy c if return_policy == True).

    Parameters:

        * cp is an instance of class consumerProblem
        * V is a NumPy array of dimension len(cp.asset_grid) x len(cp.z_vals)

    """
    # === simplify names, set up arrays === #
    R, Pi, beta, u, b = cp.R, cp.Pi, cp.beta, cp.u, cp.b  
    asset_grid, z_vals = cp.asset_grid, cp.z_vals        
    new_V = np.empty(V.shape)
    new_c = np.empty(V.shape)
    z_index = range(len(z_vals))  

    # === linear interpolation of V along the asset grid === #
    vf = lambda a, i_z: interp(a, asset_grid, V[:, i_z]) 

    # === solve r.h.s. of Bellman equation === #
    for i_a, a in enumerate(asset_grid):
        for i_z, z in enumerate(z_vals):
            def obj(c):  # objective function to be *minimized*
                y = sum(vf(R * a + z - c, j) * Pi[i_z, j] for j in z_index)
                return - u(c) - beta * y
            c_star = fminbound(obj, np.min(z_vals), R * a + z + b)
            new_c[i_a, i_z], new_V[i_a, i_z] = c_star, -obj(c_star)

    if return_policy:
        return new_c
    else:
        return new_V


def coleman_operator(cp, c):
    """
    The approximate Coleman operator.  Iteration with this operator
    corresponds to policy function iteration.  Computes and returns the
    updated consumption policy c.

    Parameters:

        * cp is an instance of class consumerProblem
        * c is a NumPy array of dimension len(cp.asset_grid) x len(cp.z_vals)

    The array c is replaced with a function cf that implements univariate
    linear interpolation over the asset grid for each possible value of z.
    """
    # === simplify names, set up arrays === #
    R, Pi, beta, du, b = cp.R, cp.Pi, cp.beta, cp.du, cp.b  
    asset_grid, z_vals = cp.asset_grid, cp.z_vals          
    z_size = len(z_vals)
    gamma = R * beta
    vals = np.empty(z_size)  

    # === linear interpolation to get consumption function === #
    def cf(a):
        """
        The call cf(a) returns an array containing the values c(a, z) for each
        z in z_vals.  For each such z, the value c(a, z) is constructed by
        univariate linear approximation over asset space, based on the values
        in the array c
        """
        for i in range(z_size):
            vals[i] = interp(a, cp.asset_grid, c[:, i])
        return vals

    # === solve for root to get Kc === #
    Kc = np.empty(c.shape)
    for i_a, a in enumerate(asset_grid):
        for i_z, z in enumerate(z_vals):
            def h(t):
                expectation = np.dot(du(cf(R * a + z - t)), Pi[i_z, :])
                return du(t) - max(gamma * expectation, du(R * a + z + b))
            Kc[i_a, i_z] = brentq(h, np.min(z_vals), R * a + z + b)

    return Kc

def initialize(cp):
    """
    Creates a suitable initial conditions V and c for value function and
    policy function iteration respectively.

        * cp is an instance of class consumerProblem.

    """
    # === simplify names, set up arrays === #
    R, beta, u, b = cp.R, cp.beta, cp.u, cp.b             
    asset_grid, z_vals = cp.asset_grid, cp.z_vals        
    shape = len(asset_grid), len(z_vals)         
    V, c = np.empty(shape), np.empty(shape)

    # === populate V and c === #
    for i_a, a in enumerate(asset_grid):
        for i_z, z in enumerate(z_vals):
            c_max = R * a + z + b
            c[i_a, i_z] = c_max
            V[i_a, i_z] = u(c_max) / (1 - beta)
    return V, c


