"""
Filename: jv.py

Authors: Thomas Sargent, John Stachurski

References
-----------

    http://quant-econ.net/jv.html

"""

import numpy as np
from scipy.integrate import fixed_quad as integrate
from scipy.optimize import minimize
import scipy.stats as stats
from scipy import interp

epsilon = 1e-4  # A small number, used in the optimization routine


class JvWorker:
    r"""
    A Jovanovic-type model of employment with on-the-job search. The
    value function is given by

    .. math::

        V(x) = \max_{\phi, s} w(x, \phi, s)

    for

    .. math::

        w(x, \phi, s) := x(1 - \phi - s)
                        + \beta (1 - \pi(s)) V(G(x, \phi))
                        + \beta \pi(s) E V[ \max(G(x, \phi), U)]

    Here

    * x = human capital
    * s = search effort
    * :math:`\phi` = investment in human capital
    * :math:`\pi(s)` = probability of new offer given search level s
    * :math:`x(1 - \phi - s)` = wage
    * :math:`G(x, \phi)` = new human capital when current job retained
    * U = RV with distribution F -- new draw of human capital

    Parameters
    ----------
    A : scalar(float), optional(default=1.4)
        Parameter in human capital transition function
    alpha : scalar(float), optional(default=0.6)
        Parameter in human capital transition function
    beta : scalar(float), optional(default=0.96)
        Discount factor
    grid_size : scalar(int), optional(default=50)
        Grid size for discretization

    Attributes
    ----------
    A : scalar(float)
        Parameter in human capital transition function
    alpha : scalar(float)
        Parameter in human capital transition function
    beta : scalar(float)
        Discount factor
    x_grid : array_like(float)
        The grid over the human capital

    """

    def __init__(self, A=1.4, alpha=0.6, beta=0.96, grid_size=50):
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

    def bellman_operator(self, V, brute_force=False, return_policies=False):
        """
        Returns the approximate value function TV by applying the
        Bellman operator associated with the model to the function V.

        Returns TV, or the V-greedy policies s_policy and phi_policy when
        return_policies=True.  In the function, the array V is replaced below
        with a function Vf that implements linear interpolation over the
        points (V(x), x) for x in x_grid.


        Parameters
        ----------
        V : array_like(float)
            Array representing an approximate value function
        brute_force : bool, optional(default=False)
            Default is False. If the brute_force flag is True, then grid
            search is performed at each maximization step.
        return_policies : bool, optional(default=False)
            Indicates whether to return just the updated value function
            TV or both the greedy policy computed from V and TV


        Returns
        -------
        s_policy : array_like(float)
            The greedy policy computed from V.  Only returned if
            return_policies == True
        new_V : array_like(float)
            The updated value function Tv, as an array representing the
            values TV(x) over x in x_grid.

        """
        # === simplify names, set up arrays, etc. === #
        G, pi, F, beta = self.G, self.pi, self.F, self.beta
        Vf = lambda x: interp(x, self.x_grid, V)
        N = len(self.x_grid)
        new_V, s_policy, phi_policy = np.empty(N), np.empty(N), np.empty(N)
        a, b = F.ppf(0.005), F.ppf(0.995)  # Quantiles, for integration
        c1 = lambda z: 1.0 - sum(z)          # used to enforce s + phi <= 1
        c2 = lambda z: z[0] - epsilon      # used to enforce s >= epsilon
        c3 = lambda z: z[1] - epsilon      # used to enforce phi >= epsilon
        guess = (0.2, 0.2)
        constraints = [{"type": "ineq", "fun": i} for i in [c1, c2, c3]]

        # === solve r.h.s. of Bellman equation === #
        for i, x in enumerate(self.x_grid):

            # === set up objective function === #
            def w(z):
                s, phi = z
                h = lambda u: Vf(np.maximum(G(x, phi), u)) * F.pdf(u)
                integral, err = integrate(h, a, b)
                q = pi(s) * integral + (1.0 - pi(s)) * Vf(G(x, phi))
                # == minus because we minimize == #
                return - x * (1.0 - phi - s) - beta * q

            # === either use SciPy solver === #
            if not brute_force:
                max_s, max_phi = minimize(w, guess, constraints=constraints,
                                          options={"disp": 0},
                                          method="COBYLA")["x"]
                max_val = -w((max_s, max_phi))

            # === or search on a grid === #
            else:
                search_grid = np.linspace(epsilon, 1.0, 15)
                max_val = -1.0
                for s in search_grid:
                    for phi in search_grid:
                        current_val = -w((s, phi)) if s + phi <= 1.0 else -1.0
                        if current_val > max_val:
                            max_val, max_s, max_phi = current_val, s, phi

            # === store results === #
            new_V[i] = max_val
            s_policy[i], phi_policy[i] = max_s, max_phi

        if return_policies:
            return s_policy, phi_policy
        else:
            return new_V
