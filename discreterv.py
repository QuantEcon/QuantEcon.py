"""
File:   discreterv.py
Author: John Stachurski, with Thomas J. Sargent
Date:   2/2013
"""

from numpy import cumsum
from numpy.random import uniform

class discreteRV:
    """
    Generates an array of draws from a discrete random variable with vector of
    probabilities given by q.  In particular, the draw() method returns i with
    probability q[i].
    """

    def __init__(self, q):
        self.set_q(q)  # q must be array like

    def set_q(self, q):
        self.Q = cumsum(q)  # Cumulative sum

    def draw(self, n=1):
        "Returns n draws from q."
        return self.Q.searchsorted(uniform(0, 1, size=n)) 


