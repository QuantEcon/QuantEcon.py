"""
Filename: ecdf.py
Authors: Thomas Sargent, John Stachurski 

Implements the empirical cumulative distribution function given an array of
observations.
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

