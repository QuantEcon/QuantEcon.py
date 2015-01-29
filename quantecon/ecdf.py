"""
Filename: ecdf.py

Authors: Thomas Sargent, John Stachurski

Implements the empirical cumulative distribution function given an array
of observations.

"""

import numpy as np


class ECDF(object):
    """
    One-dimensional empirical distribution function given a vector of
    observations.

    Parameters
    ----------
    observations : array_like
        An array of observations

    Attributes
    ----------
    observations : see Parameters

    """

    def __init__(self, observations):
        self.observations = np.asarray(observations)

    def __repr__(self):
        return "ECDF with {n} observations".format(n=self.observations.size)

    def __str__(self):
        return self.__repr__()

    def __call__(self, x):
        """
        Evaluates the ecdf at x

        Parameters
        ----------
        x : scalar(float)
            The x at which the ecdf is evaluated

        Returns
        -------
        scalar(float)
            Fraction of the sample less than x

        """
        return np.mean(self.observations <= x)
