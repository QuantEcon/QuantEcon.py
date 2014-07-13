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
        Parameters
        ----------
        q : array_like
            Nonnegative numbers that sum to 1
        """
        self._q = q
        self.Q = cumsum(q)

    def get_q(self):
        """
        Getter method for q.
        """
        return self._q

    def set_q(self, val):
        """
        Setter method for q.
        """
        self._q = val
        self.Q = cumsum(val)

    q = property(get_q, set_q)

    def draw(self, k=1):
        """
        Returns k draws from q. 
        
        For each such draw, the value i is returned with probability q[i].  

        Paramterers
        -----------
        k : int, optional
            Number of draws to be returned

        Returns
        -------
        np.ndarray
            An array of k independent draws from q
        """
        return self.Q.searchsorted(uniform(0, 1, size=k)) 


