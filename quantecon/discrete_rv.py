"""
Filename: discrete_rv.py

Authors: Thomas Sargent, John Stachurski

Generates an array of draws from a discrete random variable with a
specified vector of probabilities.

"""

from numpy import cumsum
from numpy.random import uniform


class DiscreteRV(object):
    """
    Generates an array of draws from a discrete random variable with
    vector of probabilities given by q.

    Parameters
    ----------
    q : array_like(float)
        Nonnegative numbers that sum to 1

    Attributes
    ----------
    q : see Parameters
    Q : array_like(float)
        The cumulative sum of q

    """

    def __init__(self, q):
        self._q = q
        self.Q = cumsum(q)

    def __repr__(self):
        return "DiscreteRV with {n} elements".format(n=self._q.size)

    def __str__(self):
        return self.__repr__()

    @property
    def q(self):
        """
        Getter method for q.

        """
        return self._q

    @q.setter
    def q(self, val):
        """
        Setter method for q.

        """
        self._q = val
        self.Q = cumsum(val)

    def draw(self, k=1):
        """
        Returns k draws from q.

        For each such draw, the value i is returned with probability
        q[i].

        Parameters
        -----------
        k : scalar(int), optional
            Number of draws to be returned

        Returns
        -------
        array_like(int)
            An array of k independent draws from q

        """
        return self.Q.searchsorted(uniform(0, 1, size=k))
