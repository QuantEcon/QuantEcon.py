"""
Filename: ecdf.py
Authors: Thomas Sargent, John Stachurski 

Implements the empirical cumulative distribution function given an array of
observations.
"""

import numpy as np
import matplotlib.pyplot as plt

class ECDF:
    """
    One-dimensional empirical distribution function given a vector of
    observations.

    """

    def __init__(self, observations):
        """
        Parameters
        ----------
        observations : array_like
            An array of observations 

        """
        self.observations = np.asarray(observations)

    def __call__(self, x): 
        """
        Evaluates the ecdf at x

        Parameters
        ----------
        x : scalar
            The x at which the ecdf is evaluated

        Returns
        -------
        float
            Fraction of the sample less than x

        """
        return np.mean(self.observations <= x)

    def plot(self, a=None, b=None): 
        """
        Plot the ecdf on the interval [a, b].

        Parameters
        ----------
        a : scalar, optional
            Lower end point of the plot interval
        b : scalar, optional
            Upper end point of the plot interval

        """


        # === choose reasonable interval if [a, b] not specified === #
        if not a:
            a = self.observations.min() - self.observations.std()
        if not b:
            b = self.observations.max() + self.observations.std()

        # === generate plot === #
        x_vals = np.linspace(a, b, num=100)
        f = np.vectorize(self.__call__)
        plt.plot(x_vals, f(x_vals))
        plt.show()

