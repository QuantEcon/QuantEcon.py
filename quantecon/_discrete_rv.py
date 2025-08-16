"""
Generates an array of draws from a discrete random variable with a
specified vector of probabilities.

"""

import numpy as np
from .util import check_random_state


class DiscreteRV:
    """
    Generates an array of draws from a discrete random variable with
    vector of probabilities given by q.

    Parameters
    ----------
    q : array_like(float)
        Nonnegative numbers that sum to 1.

    Attributes
    ----------
    q : see Parameters.
    Q : array_like(float)
        The cumulative sum of q.

    """

    def __init__(self, q):
        self._q = np.asarray(q)
        self.Q = np.cumsum(q)

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
        self._q = np.asarray(val)
        self.Q = np.cumsum(val)

    def draw(self, k=1, random_state=None):
        """
        Returns k draws from q.

        For each such draw, the value i is returned with probability
        q[i].

        Parameters
        ----------
        k : scalar(int), optional
            Number of draws to be returned

        random_state : int or np.random.RandomState/Generator, optional
            Random seed (integer) or np.random.RandomState or Generator
            instance to set the initial state of the random number
            generator for reproducibility. If None, a randomly
            initialized RandomState is used.

        Returns
        -------
        array_like(int)
            An array of k independent draws from q

        """
        random_state = check_random_state(random_state)

        return self.Q.searchsorted(random_state.uniform(0, 1, size=k),
                                   side='right')
