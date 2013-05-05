"""
Origin: QEwP by John Stachurski and Thomas J. Sargent
Date:   2/2013
File:   kalman.py

Implements the Kalman filter for the state space model

    x_{t+1} = A x_t + w_{t+1}
    y_t = G x_t + v_t.

Here x_t is the hidden state and y_t is the measurement.  The shocks {w_t} 
and {v_t} are iid zero mean Gaussians with covariance matrices Q and R
respectively.
"""

import numpy as np
from numpy import dot
from numpy.linalg import inv

class Kalman:

    def __init__(self, A, G, Q, R):
        """
        Provide initial parameters describing the model.  All arguments should
        be Python scalars or NumPy ndarrays.

            * A is n x n
            * Q is n x n and positive definite
            * G is k x n
            * R is k x k and positive definite
        """
        self.A = np.array(A, dtype='float32')
        self.G = np.array(G, dtype='float32')
        self.Q = np.array(Q, dtype='float32')
        self.R = np.array(R, dtype='float32')

    def set_state(self, x_hat, Sigma):
        """
        Set the state, which is the mean x_hat and covariance matrix Sigma of
        the prior/predictive density.  

            * x_hat is n x 1
            * Sigma is n x n and positive definite

        Must be Python scalars or NumPy arrays.
        """
        self.current_Sigma = np.array(Sigma, dtype='float32')
        self.current_x_hat = np.array(x_hat, dtype='float32')

    def prior_to_filtered(self, y):
        """
        Updates the moments (x_hat, Sigma) of the time t prior to the time t
        filtering distribution, using current measurement y_t.  The parameter
        y should be a Python scalar or NumPy array.  The updates are according
        to 

            x_hat^F = x_hat + Sigma G' (G Sigma G' + R)^{-1}(y - G x_hat)
            Sigma^F = Sigma - Sigma G' (G Sigma G' + R)^{-1} G Sigma

        """
        # Simplify notation
        G, R = self.G, self.R
        x_hat, Sigma = self.current_x_hat, self.current_Sigma
        # And then update
        A = dot(Sigma, G.T)
        B = dot(dot(G, Sigma), G.T) + R
        if B.shape:  # If B has a shape, then it is multidimensional
            M = dot(A, inv(B))
        else:  # Otherwise it's just scalar
            M = A / B
        self.current_x_hat = x_hat + dot(M, (y - dot(G, x_hat)))
        self.current_Sigma = Sigma  - dot(M, dot(G,  Sigma))

    def filtered_to_forecast(self):
        """
        Updates the moments of the time t filtering distribution to the
        moments of the predictive distribution -- which becomes the time t+1
        prior
        """
        # Make local copies of names to simplify notation
        A, Q = self.A, self.Q
        x_hat, Sigma = self.current_x_hat, self.current_Sigma
        # And then update
        self.current_x_hat = dot(A, x_hat)
        self.current_Sigma = dot(A, dot(Sigma, A.T)) + Q

    def update(self, y):
        """
        Updates x_hat and Sigma given k x 1 ndarray y.  The full update, from
        one period to the next
        """
        self.prior_to_filtered(y)
        self.filtered_to_forecast()

        



