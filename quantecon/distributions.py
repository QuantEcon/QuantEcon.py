"""
Probability distributions useful in economics.

References
----------

http://en.wikipedia.org/wiki/Beta-binomial_distribution

"""
from math import sqrt
import numpy as np
from scipy.special import binom, beta


class BetaBinomial:
    """
    The Beta-Binomial distribution

    Parameters
    ----------
    n : scalar(int)
        First parameter to the Beta-binomial distribution
    a : scalar(float)
        Second parameter to the Beta-binomial distribution
    b : scalar(float)
        Third parameter to the Beta-binomial distribution

    Attributes
    ----------
    n, a, b : see Parameters

    """

    def __init__(self, n, a, b):
        self.n, self.a, self.b = n, a, b

    @property
    def mean(self):
        "mean"
        n, a, b = self.n, self.a, self.b
        return n * a / (a + b)

    @property
    def std(self):
        "standard deviation"
        return sqrt(self.var)

    @property
    def var(self):
        "Variance"
        n, a, b = self.n, self.a, self.b
        top = n*a*b * (a + b + n)
        btm = (a+b)**2.0 * (a+b+1.0)
        return top / btm

    @property
    def skew(self):
        "skewness"
        n, a, b = self.n, self.a, self.b
        t1 = (a+b+2*n) * (b - a) / (a+b+2)
        t2 = sqrt((1+a+b) / (n*a*b * (n+a+b)))
        return t1 * t2

    def pdf(self):
        r"""
        Generate the vector of probabilities for the Beta-binomial
        (n, a, b) distribution.

        The Beta-binomial distribution takes the form

        .. math::
            p(k \,|\, n, a, b) =
            {n \choose k} \frac{B(k + a, n - k + b)}{B(a, b)},
            \qquad k = 0, \ldots, n,

        where :math:`B` is the beta function.

        Parameters
        ----------
        n : scalar(int)
            First parameter to the Beta-binomial distribution
        a : scalar(float)
            Second parameter to the Beta-binomial distribution
        b : scalar(float)
            Third parameter to the Beta-binomial distribution

        Returns
        -------
        probs: array_like(float)
            Vector of probabilities over k

        """
        n, a, b = self.n, self.a, self.b
        k = np.arange(n + 1)
        probs = binom(n, k) * beta(k + a, n - k + b) / beta(a, b)
        return probs

    # def cdf(self):
    #     r"""
    #     Generate the vector of cumulative probabilities for the
    #     Beta-binomial(n, a, b) distribution.

    #     The cdf of the Beta-binomial distribution takes the form

    #     .. math::
    #         P(k \,|\, n, a, b) = 1 -
    #         \frac{B(b+n-k-1, a+k+1) {}_3F_2(a,b;k)}{B(a,b) B(n-k, k+2)},
    #         \qquad k = 0, \ldots, n

    #     where :math:`B` is the beta function.

    #     Parameters
    #     ----------
    #     n : scalar(int)
    #         First parameter to the Beta-binomial distribution
    #     a : scalar(float)
    #         Second parameter to the Beta-binomial distribution
    #     b : scalar(float)
    #         Third parameter to the Beta-binomial distribution

    #     Returns
    #     -------
    #     probs: array_like(float)
    #         Vector of probabilities over k

    #     """
