"""
Filename: odu.py
Authors: Thomas Sargent, John Stachurski

Solves the "Offer Distribution Unknown" Model by value function iteration
and a second faster method discussed in the corresponding quantecon lecture.
"""
from scipy.interpolate import LinearNDInterpolator
from scipy.integrate import fixed_quad
from scipy.stats import beta as beta_distribution
from scipy import interp
from numpy import maximum as npmax
import numpy as np

class SearchProblem:
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


    def bellman_operator(self, v):
        """
        The Bellman operator.  Including for comparison. Value function
        iteration is not recommended for this problem.  See the reservation
        wage operator below.

        Parameters
        ==============
            v : An approximate value function represented as a
                one-dimensional array.

        """
        # == Simplify names == #
        f, g, beta, c, q = self.f, self.g, self.beta, self.c, self.q 

        vf = LinearNDInterpolator(self.grid_points, v)
        N = len(v)
        new_v = np.empty(N)
        for i in range(N):
            w, pi = self.grid_points[i,:]
            v1 = w / (1 - beta)
            integrand = lambda m: vf(m, q(m, pi)) * (pi * f(m) \
                    + (1 - pi) * g(m))
            integral, error = fixed_quad(integrand, 0, self.w_max)
            v2 = c + beta * integral
            new_v[i] = max(v1, v2)
        return new_v

    def get_greedy(self, v):
        """
        Compute optimal actions taking v as the value function.  Parameters
        are the same as for bellman().  Returns a NumPy array called "policy",
        where policy[i] is the optimal action at self.grid_points[i,:].  The
        optimal action is represented in binary, where 0 indicates reject and
        1 indicates accept.
        """
        # == Simplify names == #
        f, g, beta, c, q = self.f, self.g, self.beta, self.c, self.q  

        vf = LinearNDInterpolator(self.grid_points, v)
        N = len(v)
        policy = np.zeros(N, dtype=int)
        for i in range(N):
            w, pi = self.grid_points[i,:]
            v1 = w / (1 - beta)
            integrand = lambda m: vf(m, q(m, pi)) * (pi * f(m) + \
                    (1 - pi) * g(m))
            integral, error = fixed_quad(integrand, 0, self.w_max)
            v2 = c + beta * integral
            policy[i] = v1 > v2  # Evaluates to 1 or 0
        return policy


    def res_wage_operator(self, phi):
        """
        Updates the reservation wage function guess phi via the operator Q.
        Returns the updated function Q phi, represented as the array new_phi.
        
            * phi is a NumPy array with len(phi) = len(pi_grid)

        """
        # == Simplify names == #
        beta, c, f, g, q = self.beta, self.c, self.f, self.g, self.q 
        # == Turn phi into a function == #
        phi_f = lambda p: interp(p, self.pi_grid, phi)  

        new_phi = np.empty(len(phi))
        for i, pi in enumerate(self.pi_grid):
            def integrand(x):
                "Integral expression on right-hand side of operator"
                return npmax(x, phi_f(q(x,pi))) * (pi * f(x) + (1 - pi) * g(x))
            integral, error = fixed_quad(integrand, 0, self.w_max)
            new_phi[i] = (1 - beta) * c + beta * integral
        return new_phi

