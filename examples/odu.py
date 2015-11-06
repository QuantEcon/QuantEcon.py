"""
Filename: odu.py

Authors: Thomas Sargent, John Stachurski

Solves the "Offer Distribution Unknown" Model by value function
iteration and a second faster method discussed in the corresponding
quantecon lecture.

"""
from textwrap import dedent
from scipy.interpolate import LinearNDInterpolator
from scipy.integrate import fixed_quad
from scipy.stats import beta as beta_distribution
from scipy import interp
from numpy import maximum as npmax
import numpy as np


class SearchProblem(object):
    """
    A class to store a given parameterization of the "offer distribution
    unknown" model.

    Parameters
    ----------
    beta : scalar(float), optional(default=0.95)
        The discount parameter
    c : scalar(float), optional(default=0.6)
        The unemployment compensation
    F_a : scalar(float), optional(default=1)
        First parameter of beta distribution on F
    F_b : scalar(float), optional(default=1)
        Second parameter of beta distribution on F
    G_a : scalar(float), optional(default=3)
        First parameter of beta distribution on G
    G_b : scalar(float), optional(default=1.2)
        Second parameter of beta distribution on G
    w_max : scalar(float), optional(default=2)
        Maximum wage possible
    w_grid_size : scalar(int), optional(default=40)
        Size of the grid on wages
    pi_grid_size : scalar(int), optional(default=40)
        Size of the grid on probabilities

    Attributes
    ----------
    beta, c, w_max : see Parameters
    w_grid : np.ndarray
        Grid points over wages, ndim=1
    pi_grid : np.ndarray
        Grid points over pi, ndim=1
    grid_points : np.ndarray
        Combined grid points, ndim=2
    F : scipy.stats._distn_infrastructure.rv_frozen
        Beta distribution with params (F_a, F_b), scaled by w_max
    G : scipy.stats._distn_infrastructure.rv_frozen
        Beta distribution with params (G_a, G_b), scaled by w_max
    f : function
        Density of F
    g : function
        Density of G
    pi_min : scalar(float)
        Minimum of grid over pi
    pi_max : scalar(float)
        Maximum of grid over pi
    """

    def __init__(self, beta=0.95, c=0.6, F_a=1, F_b=1, G_a=3, G_b=1.2,
                 w_max=2, w_grid_size=40, pi_grid_size=40):

        self.beta, self.c, self.w_max = beta, c, w_max
        self.F = beta_distribution(F_a, F_b, scale=w_max)
        self.G = beta_distribution(G_a, G_b, scale=w_max)
        self.f, self.g = self.F.pdf, self.G.pdf    # Density functions
        self.pi_min, self.pi_max = 1e-3, 1 - 1e-3  # Avoids instability
        self.w_grid = np.linspace(0, w_max, w_grid_size)
        self.pi_grid = np.linspace(self.pi_min, self.pi_max, pi_grid_size)
        x, y = np.meshgrid(self.w_grid, self.pi_grid)
        self.grid_points = np.column_stack((x.ravel(1), y.ravel(1)))

    def __repr__(self):
        m = "SearchProblem(beta={b}, c={c}, F_a={fa}, F_b={fb}, G_a={ga}, "
        m += "G_b={gb}, w_max={wu}, w_grid_size={wgs}, pi_grid_size={pgs})"
        fa, fb = self.F.args
        ga, gb = self.G.args
        return m.format(b=self.beta, c=self.c, fa=fa, fb=fb, ga=ga,
                        gb=gb, wu=self.w_grid.max(),
                        wgs=self.w_grid.size, pgs=self.pi_grid.size)

    def __str__(self):
        m = """\
        SearchProblem (offer distribution unknown):
         - beta (discount factor)                          : {b:g}
         - c (unemployment compensation)                   : {c}
         - F (distribution F)                              : Beta({fa}, {fb:g})
         - G (distribution G)                              : Beta({ga}, {gb:g})
         - w bounds (bounds for wage offers)               : ({wl:g}, {wu:g})
         - w grid size (number of points in grid for wage) : {wgs}
         - pi bounds (bounds for probability of dist f)    : ({pl:g}, {pu:g})
         - pi grid size (number of points in grid for pi)  : {pgs}
        """
        fa, fb = self.F.args
        ga, gb = self.G.args
        return dedent(m.format(b=self.beta, c=self.c, fa=fa, fb=fb, ga=ga,
                               gb=gb,
                               wl=self.w_grid.min(), wu=self.w_grid.max(),
                               wgs=self.w_grid.size,
                               pl=self.pi_grid.min(), pu=self.pi_grid.max(),
                               pgs=self.pi_grid.size))

    def q(self, w, pi):
        """
        Updates pi using Bayes' rule and the current wage observation w.

        Returns
        -------

        new_pi : scalar(float)
            The updated probability

        """

        new_pi = 1.0 / (1 + ((1 - pi) * self.g(w)) / (pi * self.f(w)))

        # Return new_pi when in [pi_min, pi_max] and else end points
        new_pi = np.maximum(np.minimum(new_pi, self.pi_max), self.pi_min)

        return new_pi

    def bellman_operator(self, v):
        """

        The Bellman operator.  Including for comparison. Value function
        iteration is not recommended for this problem.  See the
        reservation wage operator below.

        Parameters
        ----------
        v : array_like(float, ndim=1, length=len(pi_grid))
            An approximate value function represented as a
            one-dimensional array.

        Returns
        -------
        new_v : array_like(float, ndim=1, length=len(pi_grid))
            The updated value function

        """
        # == Simplify names == #
        f, g, beta, c, q = self.f, self.g, self.beta, self.c, self.q

        vf = LinearNDInterpolator(self.grid_points, v)
        N = len(v)
        new_v = np.empty(N)

        for i in range(N):
            w, pi = self.grid_points[i, :]
            v1 = w / (1 - beta)
            integrand = lambda m: vf(m, q(m, pi)) * (pi * f(m)
                                                     + (1 - pi) * g(m))
            integral, error = fixed_quad(integrand, 0, self.w_max)
            v2 = c + beta * integral
            new_v[i] = max(v1, v2)

        return new_v

    def get_greedy(self, v):
        """
        Compute optimal actions taking v as the value function.

        Parameters
        ----------
        v : array_like(float, ndim=1, length=len(pi_grid))
            An approximate value function represented as a
            one-dimensional array.

        Returns
        -------
        policy : array_like(float, ndim=1, length=len(pi_grid))
            The decision to accept or reject an offer where 1 indicates
            accept and 0 indicates reject

        """
        # == Simplify names == #
        f, g, beta, c, q = self.f, self.g, self.beta, self.c, self.q

        vf = LinearNDInterpolator(self.grid_points, v)
        N = len(v)
        policy = np.zeros(N, dtype=int)

        for i in range(N):
            w, pi = self.grid_points[i, :]
            v1 = w / (1 - beta)
            integrand = lambda m: vf(m, q(m, pi)) * (pi * f(m) +
                                                     (1 - pi) * g(m))
            integral, error = fixed_quad(integrand, 0, self.w_max)
            v2 = c + beta * integral
            policy[i] = v1 > v2  # Evaluates to 1 or 0

        return policy

    def res_wage_operator(self, phi):
        """

        Updates the reservation wage function guess phi via the operator
        Q.

        Parameters
        ----------
        phi : array_like(float, ndim=1, length=len(pi_grid))
            This is reservation wage guess

        Returns
        -------
        new_phi : array_like(float, ndim=1, length=len(pi_grid))
            The updated reservation wage guess.

        """
        # == Simplify names == #
        beta, c, f, g, q = self.beta, self.c, self.f, self.g, self.q
        # == Turn phi into a function == #
        phi_f = lambda p: interp(p, self.pi_grid, phi)

        new_phi = np.empty(len(phi))
        for i, pi in enumerate(self.pi_grid):
            def integrand(x):
                "Integral expression on right-hand side of operator"
                return npmax(x, phi_f(q(x, pi))) * (pi*f(x) + (1 - pi)*g(x))
            integral, error = fixed_quad(integrand, 0, self.w_max)
            new_phi[i] = (1 - beta) * c + beta * integral

        return new_phi
