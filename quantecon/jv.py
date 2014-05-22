"""
Filename: jv.py
Authors: Thomas Sargent, John Stachurski 

A Jovanovic-type model of employment with on-the-job search. The value
function is given by

  V(x) = max_{phi, s} w(x, phi, s)

  for w(x, phi, s) := x(1 - phi - s) 
                        + beta (1 - pi(s)) V(G(x, phi)) 
                        + beta pi(s) E V[ max(G(x, phi), U)]
Here

    * x = human capital
    * s = search effort
    * phi = investment in human capital 
    * pi(s) = probability of new offer given search level s
    * x(1 - phi - s) = wage
    * G(x, phi) = updated human capital when current job retained
    * U = a random variable with distribution F -- new draw of human capital

"""

import numpy as np
from scipy.integrate import fixed_quad as integrate
from scipy.optimize import fmin_slsqp as minimize
import scipy.stats as stats
from scipy import interp

epsilon = 1e-4  #  A small number, used in the optimization routine

class workerProblem:

    def __init__(self, A=1.4, alpha=0.6, beta=0.96, grid_size=50):
        """
        This class is just a "struct" to hold the attributes of a given model.
        """
        self.A, self.alpha, self.beta = A, alpha, beta
        # === set defaults for G, pi and F === #
        self.G = lambda x, phi: A * (x * phi)**alpha 
        self.pi = np.sqrt 
        self.F = stats.beta(2, 2)  
        # === Set up grid over the state space for DP === #
        # Max of grid is the max of a large quantile value for F and the 
        # fixed point y = G(y, 1).
        grid_max = max(A**(1 / (1 - alpha)), self.F.ppf(1 - epsilon))
        self.x_grid = np.linspace(epsilon, grid_max, grid_size)


def bellman_operator(wp, V, brute_force=False, return_policies=False):
    """
    Parameter wp is an instance of workerProblem.  Thus function returns the
    approximate value function TV by applying the Bellman operator associated
    with the model wp to the function V.  Returns TV, or the V-greedy policies
    s_policy and phi_policy when return_policies=True.

    In the function, the array V is replaced below with a function Vf that
    implements linear interpolation over the points (V(x), x) for x in x_grid.
    If the brute_force flag is true, then grid search is performed at each
    maximization step.  In either case, T returns a NumPy array representing
    the updated values TV(x) over x in x_grid.

    """
    # === simplify names, set up arrays, etc. === #
    G, pi, F, beta = wp.G, wp.pi, wp.F, wp.beta  
    Vf = lambda x: interp(x, wp.x_grid, V) 
    N = len(wp.x_grid)
    new_V, s_policy, phi_policy = np.empty(N), np.empty(N), np.empty(N)
    a, b = F.ppf(0.005), F.ppf(0.995)  # Quantiles, for integration
    c1 = lambda z: 1 - sum(z)          # used to enforce s + phi <= 1
    c2 = lambda z: z[0] - epsilon      # used to enforce s >= epsilon
    c3 = lambda z: z[1] - epsilon      # used to enforce phi >= epsilon
    guess, constraints = (0.2, 0.2), [c1, c2, c3]

    # === solve r.h.s. of Bellman equation === #
    for i, x in enumerate(wp.x_grid):

        # === set up objective function === #
        def w(z):  
            s, phi = z
            integrand = lambda u: Vf(np.maximum(G(x, phi), u)) * F.pdf(u)
            integral, err = integrate(integrand, a, b)
            q = pi(s) * integral + (1 - pi(s)) * Vf(G(x, phi))
            return - x * (1 - phi - s) - beta * q  # minus because we minimize

        # === either use SciPy solver === #
        if not brute_force:  
            max_s, max_phi = minimize(w, guess, ieqcons=constraints, disp=0)
            max_val = -w((max_s, max_phi))

        # === or search on a grid === #
        else:  
            search_grid = np.linspace(epsilon, 1, 15)
            max_val = -1
            for s in search_grid:
                for phi in search_grid:
                    current_val = -w((s, phi)) if s + phi <= 1 else -1
                    if current_val > max_val:
                        max_val, max_s, max_phi = current_val, s, phi

        # === store results === #
        new_V[i] = max_val
        s_policy[i], phi_policy[i] = max_s, max_phi

    if return_policies:
        return s_policy, phi_policy
    else:
        return new_V


