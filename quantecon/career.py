"""
Filename: career.py
Authors: Thomas Sargent, John Stachurski 

A collection of functions to solve the career / job choice model due to Derek Neal.

References
----------

  http://quant-econ.net/career.html

..  [Neal1999] Neal, D. (1999). The Complexity of Job Mobility among Young Men, 
               Journal of Labor Economics, 17(2), 237-261.

"""

import numpy as np
from scipy.special import binom, beta


def gen_probs(n, a, b):
    """
    Generate the vector of probabilities for the Beta-binomial 
    (n, a, b) distribution.

    The Beta-binomial distribution takes the form

    .. math::
        p(k \,|\, n, a, b) 
        = {n \choose k} \frac{B(k + a, n - k + b)}{B(a, b)},
        \qquad k = 0, \ldots, n

    Parameters
    ----------
    n : int
        First parameter to the Beta-binomial distribution
    a : float
        Second parameter to the Beta-binomial distribution
    b : float
        Third parameter to the Beta-binomial distribution

    Returns
    -------
    probs: np.ndarray
        Vector of probabilities over k

    """
    probs = np.zeros(n+1)
    for k in range(n+1):
        probs[k] = binom(n, k) * beta(k + a, n - k + b) / beta(a, b)
    return probs


class CareerWorkerProblem:
    """
    An instance of the class is an object with data on a particular problem of
    this type, including probabilites, discount factor and sample space for
    the variables.
    """

    def __init__(self, B=5.0, beta=0.95, N=50, F_a=1, F_b=1, G_a=1, G_b=1):
        """
        Parameters
        ----------
        beta : float, optional
            Discount factor
        B : float, optional
            Upper bound of for both epsilon and theta
        N : int, optional
            Number of possible realizations for both epsilon and theta
        F_a : int or float, optional
            Parameter `a` from the career distribution 
        F_b : int or float, optional
            Parameter `b` from the career distribution 
        G_a : int or float, optional
            Parameter `a` from the job distribution 
        G_b : int or float, optional
            Parameter `b` from the job distribution 

        """
        self.beta, self.N, self.B = beta, N, B
        self.theta = np.linspace(0, B, N)     # set of theta values
        self.epsilon = np.linspace(0, B, N)   # set of epsilon values
        self.F_probs = gen_probs(N-1, F_a, F_b)
        self.G_probs = gen_probs(N-1, G_a, G_b)
        self.F_mean = np.sum(self.theta * self.F_probs)
        self.G_mean = np.sum(self.epsilon * self.G_probs)

    def bellman(self, v):
        """
        The Bellman operator for the career / job choice model of Neal.  

        Parameters
        ----------
            v : np.ndarray
                A 2D NumPy array representing the value function
                Interpretation: :math:`v[i, j] = v(\theta_i, \epsilon_j)`

        Returns
        -------
            new_v : np.ndarray
                The updated value function Tv as an array of shape v.shape

        """
        new_v = np.empty(v.shape)
        for i in range(self.N):
            for j in range(self.N):
                v1 = self.theta[i] + self.epsilon[j] + self.beta * v[i, j]
                v2 = self.theta[i] + self.G_mean + self.beta * \
                        np.dot(v[i, :], self.G_probs)
                v3 = self.G_mean + self.F_mean + self.beta * \
                        np.dot(self.F_probs, np.dot(v, self.G_probs))
                new_v[i, j] = max(v1, v2, v3)
        return new_v

    def get_greedy(self, v):
        """
        Compute optimal actions taking v as the value function.  
        
        Parameters
        ----------
            v : np.ndarray
                A 2D NumPy array representing the value function
                Interpretation: :math:`v[i, j] = v(\theta_i, \epsilon_j)`

        Returns
        -------
            policy : np.ndarray
                A 2D NumPy array, where policy[i, j] is the optimal action 
                at :math:`(\theta_i, \epsilon_j)`.  

                The optimal action is represented as an integer in the set 1,
                2, 3, where 1 = 'stay put', 2 = 'new job' and 3 = 'new life'

        """
        policy = np.empty(v.shape, dtype=int)
        for i in range(self.N):
            for j in range(self.N):
                v1 = self.theta[i] + self.epsilon[j] + self.beta * v[i, j]
                v2 = self.theta[i] + self.G_mean + self.beta * \
                        np.dot(v[i, :], self.G_probs)
                v3 = self.G_mean + self.F_mean + self.beta * \
                        np.dot(self.F_probs, np.dot(v, self.G_probs))
                if v1 > max(v2, v3):
                    action = 1  
                elif v2 > max(v1, v3):
                    action = 2
                else:
                    action = 3
                policy[i, j] = action
        return policy
