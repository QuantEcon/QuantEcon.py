"""
Filename: career.py
Authors: Thomas Sargent, John Stachurski 

A collection of functions to solve the career / job choice model of Neal.
"""

import numpy as np
from scipy.special import binom, beta


def gen_probs(n, a, b):
    """
    Generate and return the vector of probabilities for the Beta-binomial 
    (n, a, b) distribution.
    """
    probs = np.zeros(n+1)
    for k in range(n+1):
        probs[k] = binom(n, k) * beta(k + a, n - k + b) / beta(a, b)
    return probs


class CareerWorkerProblem:

    def __init__(self, B=5.0, beta=0.95, N=50, F_a=1, F_b=1, G_a=1, G_b=1):
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
        
            * v is a 2D NumPy array representing the value function
            
        The array v should be interpreted as 
        
            v[i, j] = v(theta_i, epsilon_j).  

        Returns updated value function Tv as an array of shape v.shape
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
        Compute optimal actions taking v as the value function.  Parameters 
        are the same as for bellman().  Returns a 2D NumPy array "policy", 
        where policy[i, j] is the optimal action at (theta_i, epsilon_j).  

        The optimal action is represented as an integer in the set 1, 2, 3, 
        where 1 = 'stay put', 2 = 'new job' and 3 = 'new life'
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
