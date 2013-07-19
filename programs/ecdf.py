"""
Origin: QEwP by John Stachurski and Thomas J. Sargent
Date:   5/2013
File:   ecdf.py

Implements the empirical distribution function.

"""

import numpy as np
import matplotlib.pyplot as plt

class ecdf:

    def __init__(self, observations):
        self.observations = np.asarray(observations)

    def __call__(self, x): 
        return np.mean(self.observations <= x)

    def plot(self, a=None, b=None): 

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

