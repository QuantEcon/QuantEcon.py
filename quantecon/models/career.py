"""
Filename: career.py

Authors: Thomas Sargent, John Stachurski

A class to solve the career / job choice model due to Derek Neal.

References
----------

http://quant-econ.net/career.html

..  [Neal1999] Neal, D. (1999). The Complexity of Job Mobility among
    Young Men, Journal of Labor Economics, 17(2), 237-261.

"""

import numpy as np
from quantecon.distributions import BetaBinomial


class CareerWorkerProblem(object):
    """
    An instance of the class is an object with data on a particular
    problem of this type, including probabilites, discount factor and
    sample space for the variables.

    Parameters
    ----------
    beta : scalar(float), optional(default=5.0)
        Discount factor
    B : scalar(float), optional(default=0.95)
        Upper bound of for both epsilon and theta
    N : scalar(int), optional(default=50)
        Number of possible realizations for both epsilon and theta
    F_a : scalar(int or float), optional(default=1)
        Parameter `a` from the career distribution
    F_b : scalar(int or float), optional(default=1)
        Parameter `b` from the career distribution
    G_a : scalar(int or float), optional(default=1)
        Parameter `a` from the job distribution
    G_b : scalar(int or float), optional(default=1)
        Parameter `b` from the job distribution

    Attributes
    ----------
    beta : scalar(float)
        Discount factor
    B : scalar(float)
        Upper bound of for both epsilon and theta
    N : scalar(int)
        Number of possible realizations for both epsilon and theta
    theta : array_like(float, ndim=1)
        A grid of values from 0 to B
    epsilon : array_like(float, ndim=1)
        A grid of values from 0 to B
    F_probs : array_like(float, ndim=1)
        The probabilities of different values for F
    G_probs : array_like(float, ndim=1)
        The probabilities of different values for G
    F_mean : scalar(float)
        The mean of the distribution for F
    G_mean : scalar(float)
        The mean of the distribution for G

    """

    def __init__(self, B=5.0, beta=0.95, N=50, F_a=1, F_b=1, G_a=1,
                 G_b=1):
        self.beta, self.N, self.B = beta, N, B
        self.theta = np.linspace(0, B, N)     # set of theta values
        self.epsilon = np.linspace(0, B, N)   # set of epsilon values
        self.F_probs = BetaBinomial(N-1, F_a, F_b).pdf()
        self.G_probs = BetaBinomial(N-1, G_a, G_b).pdf()
        self.F_mean = np.sum(self.theta * self.F_probs)
        self.G_mean = np.sum(self.epsilon * self.G_probs)

    def bellman(self, v):
        """
        The Bellman operator for the career / job choice model of Neal.

        Parameters
        ----------
        v : array_like(float)
            A 2D NumPy array representing the value function
            Interpretation: :math:`v[i, j] = v(\theta_i, \epsilon_j)`

        Returns
        -------
        new_v : array_like(float)
            The updated value function Tv as an array of shape v.shape

        """
        new_v = np.empty(v.shape)
        for i in range(self.N):
            for j in range(self.N):
                # stay put
                v1 = self.theta[i] + self.epsilon[j] + self.beta * v[i, j]

                # new job
                v2 = (self.theta[i] + self.G_mean + self.beta *
                      np.dot(v[i, :], self.G_probs))

                # new life
                v3 = (self.G_mean + self.F_mean + self.beta *
                      np.dot(self.F_probs, np.dot(v, self.G_probs)))
                new_v[i, j] = max(v1, v2, v3)
        return new_v

    def get_greedy(self, v):
        """
        Compute optimal actions taking v as the value function.

        Parameters
        ----------
        v : array_like(float)
            A 2D NumPy array representing the value function
            Interpretation: :math:`v[i, j] = v(\theta_i, \epsilon_j)`

        Returns
        -------
        policy : array_like(float)
            A 2D NumPy array, where policy[i, j] is the optimal action
            at :math:`(\theta_i, \epsilon_j)`.

            The optimal action is represented as an integer in the set
            1, 2, 3, where 1 = 'stay put', 2 = 'new job' and 3 = 'new
            life'

        """
        policy = np.empty(v.shape, dtype=int)
        for i in range(self.N):
            for j in range(self.N):
                v1 = self.theta[i] + self.epsilon[j] + self.beta * v[i, j]
                v2 = (self.theta[i] + self.G_mean + self.beta *
                      np.dot(v[i, :], self.G_probs))
                v3 = (self.G_mean + self.F_mean + self.beta *
                      np.dot(self.F_probs, np.dot(v, self.G_probs)))
                if v1 > max(v2, v3):
                    action = 1
                elif v2 > max(v1, v3):
                    action = 2
                else:
                    action = 3
                policy[i, j] = action

        return policy
