"""
Origin: QE by John Stachurski and Thomas J. Sargent
Filename: odu_vfi.py
Authors: John Stachurski and Thomas Sargent
LastModified: 11/08/2013

Solves the "Offer Distribution Unknown" Model by value function iteration.

Note that a much better technique is given in solution_odu_ex1.py
"""
from scipy.interpolate import LinearNDInterpolator
from scipy.integrate import fixed_quad
from scipy.stats import beta as beta_distribution
import numpy as np

class searchProblem:
    """
    A class to store a given parameterization of the "offer distribution
    unknown" model.
    """

    def __init__(self, beta=0.95, c=0.6, F_a=1, F_b=1, G_a=3, G_b=1.2, 
            w_max=2, w_grid_size=40, pi_grid_size=40):
        """
        Sets up parameters and grid.  The attribute "grid_points" defined
        below is a 2 column array that stores the 2D grid points for the DP
        problem. Each row represents a single (w, pi) pair.
        """
        self.beta, self.c, self.w_max = beta, c, w_max
        self.F = beta_distribution(F_a, F_b, scale=w_max)
        self.G = beta_distribution(G_a, G_b, scale=w_max)
        self.f, self.g = self.F.pdf, self.G.pdf    # Density functions
        self.pi_min, self.pi_max = 1e-3, 1 - 1e-3  # Avoids instability
        self.w_grid = np.linspace(0, w_max, w_grid_size)
        self.pi_grid = np.linspace(self.pi_min, self.pi_max, pi_grid_size)
        x, y = np.meshgrid(self.w_grid, self.pi_grid)
        self.grid_points = np.column_stack((x.ravel(1), y.ravel(1)))

    def q(self, w, pi):
        """
        Updates pi using Bayes' rule and the current wage observation w.
        """
        new_pi = 1.0 / (1 + ((1 - pi) * self.g(w)) / (pi * self.f(w)))
        # Return new_pi when in [pi_min, pi_max], and the end points otherwise
        return np.maximum(np.minimum(new_pi, self.pi_max), self.pi_min)


def bellman(sp, v):
    """
    The Bellman operator.

        * sp is an instance of searchProblem
        * v is an approximate value function represented as a one-dimensional
            array.
    """
    f, g, beta, c, q = sp.f, sp.g, sp.beta, sp.c, sp.q  # Simplify names
    vf = LinearNDInterpolator(sp.grid_points, v)
    N = len(v)
    new_v = np.empty(N)
    for i in range(N):
        w, pi = sp.grid_points[i,:]
        v1 = w / (1 - beta)
        integrand = lambda m: vf(m, q(m, pi)) * (pi * f(m) + (1 - pi) * g(m))
        integral, error = fixed_quad(integrand, 0, sp.w_max)
        v2 = c + beta * integral
        new_v[i] = max(v1, v2)
    return new_v

def get_greedy(sp, v):
    """
    Compute optimal actions taking v as the value function.  Parameters are
    the same as for bellman().  Returns a NumPy array called "policy", where
    policy[i] is the optimal action at sp.grid_points[i,:].  The optimal
    action is represented in binary, where 0 indicates reject and 1 indicates
    accept.
    """
    f, g, beta, c, q = sp.f, sp.g, sp.beta, sp.c, sp.q  # Simplify names
    vf = LinearNDInterpolator(sp.grid_points, v)
    N = len(v)
    policy = np.zeros(N, dtype=int)
    for i in range(N):
        w, pi = sp.grid_points[i,:]
        v1 = w / (1 - beta)
        integrand = lambda m: vf(m, q(m, pi)) * (pi * f(m) + (1 - pi) * g(m))
        integral, error = fixed_quad(integrand, 0, sp.w_max)
        v2 = c + beta * integral
        policy[i] = v1 > v2  # Evaluates to 1 or 0
    return policy

