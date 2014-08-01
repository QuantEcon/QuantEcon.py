"""
Filename: asset_pricing.py

Authors: David Evans, John Stachurski and Thomas J. Sargent

Computes asset prices in an endowment economy when the endowment obeys
geometric growth driven by a finite state Markov chain.  The transition
matrix of the Markov chain is P, and the set of states is s.  The
discount factor is beta, and gamma is the coefficient of relative risk
aversion in the household's utility function.

References
----------

    http://quant-econ.net/markov_asset.html

"""

import numpy as np
from numpy.linalg import solve


class AssetPrices(object):
    r"""
    A class to compute asset prices when the endowment follows a finite
    Markov chain.

    Parameters
    ----------
    beta : scalar, float
        Discount factor
    P : array_like(float)
        Transition matrix
    s : array_like(float)
        Growth rate of consumption
    gamma : scalar(float)
        Coefficient of risk aversion

    Attributes
    ----------
    beta : scalar(float)
        Discount factor
    P : array_like(float)
        Transition matrix
    s : array_like(float)
        Growth rate of consumption
    gamma : scalar(float)
        Coefficient of risk aversion
    n : scalar(int)
        The number of rows in P

    Examples
    --------

    >>> n = 5
    >>> P = 0.0125 * np.ones((n, n))
    >>> P += np.diag(0.95 - 0.0125 * np.ones(5))
    >>> s = np.array([1.05, 1.025, 1.0, 0.975, 0.95])
    >>> gamma = 2.0
    >>> beta = 0.94
    >>> ap = AssetPrices(beta, P, s, gamma)
    >>> zeta = 1.0
    >>> v = ap.tree_price()
    >>> print("Lucas Tree Prices: %s" % v)
    Lucas Tree Prices: [ 12.72221763  14.72515002  17.57142236
    21.93570661  29.47401578]

    >>> v_consol = ap.consol_price(zeta)
    >>> print("Consol Bond Prices: %s" % v_consol)
    Consol Bond Prices:  [  87.56860139  109.25108965  148.67554548
    242.55144082  753.87100476]

    >>> p_s = 150.0
    >>> w_bar, w_bars = ap.call_option(zeta, p_s, T = [10,20,30])
    >>> w_bar
    array([  64.30843769,   80.05179282,  108.67734545,  176.83933585,
        603.87100476])
    >>> w_bars
    {10: array([  44.79815889,   50.81409953,   58.61386544,
         115.69837047, 603.87100476]),
     20: array([  56.73357192,   68.51905592,   86.69038119,
         138.45961867, 603.87100476]),
     30: array([  60.62653565,   74.67608505,   98.38386204,
          153.80497466, 603.87100476])}

    """
    def __init__(self, beta, P, s, gamma):
        self.beta, self.gamma = beta, gamma
        self.P, self.s = P, s
        self.n = self.P.shape[0]

    @property
    def P_tilde(self):
        P, s, gamma = self.P, self.s, self.gamma
        return P * s**(1.0-gamma)  # using broadcasting

    @property
    def P_check(self):
        P, s, gamma = self.P, self.s, self.gamma
        return P * s**(-gamma)  # using broadcasting

    def tree_price(self):
        """
        Computes the function v such that the price of the lucas tree is
        v(lambda)C_t

        Returns
        -------
        v : array_like(float)
            Lucas tree prices

        """
        # == Simplify names == #
        beta = self.beta

        # == Compute v == #
        P_tilde = self.P_tilde
        I = np.identity(self.n)
        O = np.ones(self.n)
        v = beta * solve(I - beta * P_tilde, P_tilde.dot(O))

        return v

    def consol_price(self, zeta):
        """
        Computes price of a consol bond with payoff zeta

        Parameters
        ----------
        zeta : scalar(float)
            Coupon of the console

        Returns
        -------
        p_bar : array_like(float)
            Console bond prices

        """
        # == Simplify names == #
        beta = self.beta

        # == Compute price == #
        P_check = self.P_check
        I = np.identity(self.n)
        O = np.ones(self.n)
        p_bar = beta * solve(I - beta * P_check, P_check.dot(zeta * O))

        return p_bar

    def call_option(self, zeta, p_s, T=[], epsilon=1e-8):
        """
        Computes price of a call option on a consol bond, both finite
        and infinite horizon

        Parameters
        ----------
        zeta : scalar(float)
            Coupon of the console

        p_s : scalar(float)
            Strike price

        T : iterable(integers)
            Length of option in the finite horizon case

        epsilon : scalar(float), optional(default=1e-8)
            Tolerance for infinite horizon problem

        Returns
        -------
        w_bar : array_like(float)
            Infinite horizon call option prices

        w_bars : dict
            A dictionary of key-value pairs {t: vec}, where t is one of
            the dates in the list T and vec is the option prices at that
            date

        """
        # == Simplify names, initialize variables == #
        beta = self.beta
        P_check = self.P_check

        # == Compute consol price == #
        v_bar = self.consol_price(zeta)

        # == Compute option price == #
        w_bar = np.zeros(self.n)
        error = epsilon + 1
        t = 0
        w_bars = {}
        while error > epsilon:
            if t in T:
                w_bars[t] = w_bar

            # == Maximize across columns == #
            to_stack = (beta*P_check.dot(w_bar), v_bar-p_s)
            w_bar_new = np.amax(np.vstack(to_stack), axis=0)

            # == Find maximal difference of each component == #
            error = np.amax(np.abs(w_bar-w_bar_new))

            # == Update == #
            w_bar = w_bar_new
            t += 1

        return w_bar, w_bars
