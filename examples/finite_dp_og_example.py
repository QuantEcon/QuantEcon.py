"""
A simple optimal growth model, for testing the DiscreteDP class.

Filename: finite_dp_og_example.py
"""
import numpy as np

class SimpleOG(object):

    def __init__(self, B=10, M=5, alpha=0.5, beta=0.9):
        """
        Set up R, Q and beta, the three elements that define an instance of
        the DiscreteDP class.
        """

        self.B, self.M, self.alpha, self.beta  = B, M, alpha, beta
        self.n = B + M + 1
        self.m = M + 1

        self.R = np.empty((self.n, self.m))
        self.Q = np.zeros((self.n, self.m, self.n))

        self.populate_Q()
        self.populate_R()

    def u(self, c):
        return c**self.alpha

    def populate_R(self):
        """
        Populate the R matrix, with R[s, a] = -np.inf for infeasible
        state-action pairs.
        """
        for s in range(self.n):
            for a in range(self.m):
                self.R[s, a] = self.u(s - a) if a <= s else -np.inf

    def populate_Q(self):
        """
        Populate the Q matrix by setting

            Q[s, a, s'] = 1 / (1 + B) if a <= s' <= a + B

        and zero otherwise.
        """

        for a in range(self.m):
            self.Q[:, a, a:(a + self.B + 1)] = 1.0 / (self.B + 1)

