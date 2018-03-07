"""
Filename: discrete_rv.py

Authors: Thomas Sargent, John Stachurski

Generates an array of draws from a discrete random variable with a
specified vector of probabilities.

"""

import numpy as np
from numpy import cumsum
from numpy.random import uniform
from numba import jit


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
    q : see Parameters
    Q : array_like(float)
        The cumulative sum of q.

    """

    def __init__(self, q):
        self._q = np.asarray(q)
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
        self._q = np.asarray(val)
        self.Q = cumsum(val)

    def draw(self, k=None, seed=None):
        """
        Returns k draws from q.

        For each such draw, the value i is returned with probability
        q[i].

        Parameters
        -----------
        k : scalar(int), optional
            Number of draws to be returned.
        seed : scalar(int), optional
            Random seed (integer) to set the initial state of the random number
            generator for reproducibility. If None, a seed is randomly
            generated.

        Returns
        -------
        array_like(int)
            An array of k independent draws from q.

        """
        if seed is None:
            seed = np.random.randint(10**18)

        if k is None:
            return random_choice_scalar(self._q, seed=seed, cum_sum=self.Q)
        else:
            return random_choice(self._q, seed=seed, cum_sum=self.Q, size=k)


@jit(nopython=True)
def random_choice_scalar(p_vals, seed, cum_sum=None):
    """
    Returns one draw from `p_vals`. Optimized using Numba and compilied in
    nopython mode.

    Parameters
    -----------
    p_vals : array_like(float)
        Nonnegative numbers that sum to 1.
    seed : scalar(int)
        Random seed (integer) to set the initial state of the random number
        generator for reproducibility.
    cum_sum : array_like(float), optional
        The cumulative sum of p_vals.

    Returns
    -------
    scalar(int)
       One draw from p_vals.

    """
    np.random.seed(seed)

    if cum_sum is None:
        cum_sum = cumsum(p_vals)

    return np.searchsorted(a=cum_sum, v=uniform(0, 1))


@jit(nopython=True)
def random_choice(p_vals, seed, cum_sum=None, size=1):
    """
    Returns `size` draws from `p_vals`. Optimized using Numba and compilied in
    nopython mode.

    For each such draw, the value i is returned with probability
    p_vals[i].

    Parameters
    -----------
    p_vals : array_like(float)
        Nonnegative numbers that sum to 1.
    seed : scalar(int)
        Random seed (integer) to set the initial state of the random number
        generator for reproducibility.
    cum_sum : array_like(float), optional
        The cumulative sum of p_vals.
    size : scalar(int), optional
        Number of draws to be returned.

    Returns
    -------
    array_like(int)
        An array of k independent draws from p_vals.

    """
    np.random.seed(seed)

    if cum_sum is None:
        cum_sum = cumsum(p_vals)

    return np.searchsorted(a=cum_sum, v=uniform(0, 1, size=size))
