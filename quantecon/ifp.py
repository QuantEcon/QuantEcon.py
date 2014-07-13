"""
Filename: ifp.py
Authors: Thomas Sargent, John Stachurski 

Tools for solving the standard optimal savings / income fluctuation problem
for an infinitely lived consumer facing an exogenous income process that
evolves according to a Markov chain.

References
----------

    http://quant-econ.net/ifp.html

"""

import numpy as np
from scipy.optimize import fminbound, brentq
from scipy import interp

class ConsumerProblem:
    """
    A class for solving the income fluctuation problem. Iteration with either
    the Coleman or Bellman operators from appropriate initial conditions
    leads to convergence to the optimal consumption policy.  The income
    process is a finite state Markov chain.  Note that the Coleman operator
    is the preferred method, as it is almost always faster and more accurate.
    The Bellman operator is only provided for comparison.

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
        Parameters
        ----------
        r : scalar
            A strictly positive scalar giving the interest rate
        beta : scalar
            The discount factor, must satisfy (1 + r) * beta < 1
        Pi : np.ndarray
            A 2D NumPy array giving the Markov matrix for {z_t}
        z_vals : array_like
            The state space of {z_t}
        u : callable
            The utility function 
        du : callable
            The derivative of u
        b : float
            The borrowing constraint
        grid_max : float
            Max of the grid used to solve the problem
        grid_min : float
            Min of the grid used to solve the problem

        """
        self.u, self.du = u, du
        self.r, self.R = r, 1 + r
        self.beta, self.b = beta, b
        self.Pi, self.z_vals = np.array(Pi), tuple(z_vals)
        self.asset_grid = np.linspace(-b, grid_max, grid_size)


    def bellman_operator(self, V, return_policy=False):
        """
        The approximate Bellman operator, which computes and returns the
        updated value function TV (or the V-greedy policy c if return_policy
        is True).

        Parameters
        ----------
        V : np.ndarray
            A NumPy array of dim len(cp.asset_grid) x len(cp.z_vals)
        return_policy : bool, optional
            Indicates whether to return the greed policy given V or the 
            updated value function TV.  Default is TV.

        Returns
        -------
        np.ndarray
            Returns either the greed policy given V or the updated value
            function TV.

        """
        # === Simplify names, set up arrays === #
        R, Pi, beta, u, b = self.R, self.Pi, self.beta, self.u, self.b  
        asset_grid, z_vals = self.asset_grid, self.z_vals        
        new_V = np.empty(V.shape)
        new_c = np.empty(V.shape)
        z_idx = range(len(z_vals))  

        # === Linear interpolation of V along the asset grid === #
        vf = lambda a, i_z: interp(a, asset_grid, V[:, i_z]) 

        # === Solve r.h.s. of Bellman equation === #
        for i_a, a in enumerate(asset_grid):
            for i_z, z in enumerate(z_vals):
                def obj(c):  # objective function to be *minimized*
                    y = sum(vf(R * a + z - c, j) * Pi[i_z, j] for j in z_idx)
                    return - u(c) - beta * y
                c_star = fminbound(obj, np.min(z_vals), R * a + z + b)
                new_c[i_a, i_z], new_V[i_a, i_z] = c_star, -obj(c_star)

        if return_policy:
            return new_c
        else:
            return new_V


    def coleman_operator(self, c):
        """
        The approximate Coleman operator.  
        
        Iteration with this operator corresponds to policy function iteration.
        Computes and returns the updated consumption policy c.  The array c is
        replaced with a function cf that implements univariate linear
        interpolation over the asset grid for each possible value of z.

        Parameters
        ----------
        c : np.ndarray
            A NumPy array of dim len(cp.asset_grid) x len(cp.z_vals)

        Returns
        -------
        np.ndarray
            The updated policy, where updating is by the Coleman operator.
            function TV.

        """
        # === simplify names, set up arrays === #
        R, Pi, beta, du, b = self.R, self.Pi, self.beta, self.du, self.b  
        asset_grid, z_vals = self.asset_grid, self.z_vals          
        z_size = len(z_vals)
        gamma = R * beta
        vals = np.empty(z_size)  

        # === linear interpolation to get consumption function === #
        def cf(a):
            """
            The call cf(a) returns an array containing the values c(a, z) for
            each z in z_vals.  For each such z, the value c(a, z) is
            constructed by univariate linear approximation over asset space,
            based on the values in the array c
            """
            for i in range(z_size):
                vals[i] = interp(a, asset_grid, c[:, i])
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

    def initialize(self):
        """
        Creates a suitable initial conditions V and c for value function and
        policy function iteration respectively.

        Returns
        -------
        np.ndarray : V
            Initial condition for value function iteration
        np.ndarray : c
            Initial condition for Coleman operator iteration
        """
        # === Simplify names, set up arrays === #
        R, beta, u, b = self.R, self.beta, self.u, self.b             
        asset_grid, z_vals = self.asset_grid, self.z_vals        
        shape = len(asset_grid), len(z_vals)         
        V, c = np.empty(shape), np.empty(shape)

        # === Populate V and c === #
        for i_a, a in enumerate(asset_grid):
            for i_z, z in enumerate(z_vals):
                c_max = R * a + z + b
                c[i_a, i_z] = c_max
                V[i_a, i_z] = u(c_max) / (1 - beta)
        return V, c


