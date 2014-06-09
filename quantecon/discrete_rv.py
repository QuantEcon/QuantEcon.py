"""
Filename: discrete_rv.py
Authors: Thomas Sargent, John Stachurski 

Generates an array of draws from a discrete random variable with a specified
vector of probabilities.
"""

from numpy import cumsum
from numpy.random import uniform

class DiscreteRV(object):
    """
    Generates an array of draws from a discrete random variable with vector of
    probabilities given by q.  
    """

    def __init__(self, q):
        """
        The argument q is a NumPy array, or array like, nonnegative and sums
        to 1
        """
        self._q = q
        self.Q = cumsum(q)

    def get_q(self):
        return self._q

    def set_q(self, val):
        self._q = val
        self.Q = cumsum(val)

    q = property(get_q, set_q)

    def draw(self, k=1):
        """
        Returns k draws from q. For each such draw, the value i is returned
        with probability q[i].  
        """
        return self.Q.searchsorted(uniform(0, 1, size=k)) 


