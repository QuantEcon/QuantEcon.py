"""
Filename: kalman.py

Authors: Thomas Sargent, John Stachurski

Implements the Kalman filter for a linear Gaussian state space model.

"""
from textwrap import dedent
import numpy as np
from numpy import dot
from scipy.linalg import inv
from .matrix_eqn import solve_discrete_riccati


class Kalman(object):
    r"""
    Implements the Kalman filter for the Gaussian state space model

    .. math::

        x_{t+1} &= A x_t + w_{t+1}\\
        y_t &= G x_t + v_t.

    Here :math:`x_t` is the hidden state and :math:`y_t` is the
    measurement. The shocks :math:`w_t` and :math:`v_t` are iid zero
    mean Gaussians with covariance matrices Q and R respectively.

    Parameters
    -----------
    A : array_like or scalar(float)
        The n x n matrix A
    Q : array_like or scalar(float)
        Q is n x n, symmetric and nonnegative definite
    G : array_like or scalar(float)
        G is k x n
    R : array_like or scalar(float)
        R is k x k, symmetric and nonnegative definite

    Attributes
    ----------
    A, Q, G, R : see Parameters
    k : scalar(int)
        Number of rows of G
    n : scalar(int)
        Number of columns of G
    current_Sigma : array_like or scalar(float)
        The n x n covariance matrix
    current_x_hat : array_like or scalar(float)
        The mean of the state

    References
    ----------

    http://quant-econ.net/kalman.html

    """

    def __init__(self, A, G, Q, R):
        self.A, self.G, self.Q, self.R = list(map(self.convert, (A, G, Q, R)))
        self.k, self.n = self.G.shape

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        m = """\
        Kalman filter:
          - dimension of state space          : {n}
          - dimension of observation equation : {k}
        """
        return dedent(m.format(n=self.n, k=self.k))

    def convert(self, x):
        """
        Convert array_like objects (lists of lists, floats, etc.) into
        well formed 2D NumPy arrays

        Parameters
        ----------
        x : scalar or array_like(float)
            Argument to be converted into a 2D NumPy array

        Returns
        -------
        array_like(float)
            A 2D NumPy array

        """
        return np.atleast_2d(np.asarray(x, dtype='float32'))

    def set_state(self, x_hat, Sigma):
        """
        Set the state of the filter (mean and variance of prior
        density).

        Parameters
        ----------
        x_hat : scalar(float) or array_like(float)
            An n x 1 array representing the mean x_hat and covariance
            matrix Sigma of the prior/predictive density.
        Sigma : scalar(float) or array_like(float)
            An n x n array representing the covariance matrix Sigma of
            the prior/predictive density.  Must be positive definite.

        """
        self.current_Sigma = self.convert(Sigma)
        self.current_x_hat = self.convert(x_hat)
        self.current_x_hat.shape = self.n, 1

    def prior_to_filtered(self, y):
        r"""
        Updates the moments (x_hat, Sigma) of the time t prior to the
        time t filtering distribution, using current measurement y_t.

        The updates are according to

        .. math::

            x_{hat}^F &= x_{hat} + \Sigma G' (G \Sigma G' + R)^{-1}
            (y - G x_{hat}) \\
            \Sigma^F &= \Sigma - \Sigma G' (G \Sigma G' + R)^{-1} G
            \Sigma

        Parameters
        ----------
        y : scalar or array_like(float)
            The current measurement

        """
        # === simplify notation === #
        G, R = self.G, self.R
        x_hat, Sigma = self.current_x_hat, self.current_Sigma

        # === and then update === #
        y = self.convert(y)
        y.shape = self.k, 1
        A = dot(Sigma, G.T)
        B = dot(dot(G, Sigma), G.T) + R
        M = dot(A, inv(B))
        self.current_x_hat = x_hat + dot(M, (y - dot(G, x_hat)))
        self.current_Sigma = Sigma - dot(M, dot(G,  Sigma))

    def filtered_to_forecast(self):
        """
        Updates the moments of the time t filtering distribution to the
        moments of the predictive distribution, which becomes the time
        t+1 prior

        """
        # === simplify notation === #
        A, Q = self.A, self.Q
        x_hat, Sigma = self.current_x_hat, self.current_Sigma

        # === and then update === #
        self.current_x_hat = dot(A, x_hat)
        self.current_Sigma = dot(A, dot(Sigma, A.T)) + Q

    def update(self, y):
        """
        Updates `x_hat` and `Sigma` given k x 1 ndarray `y`.  The full
        update, from one period to the next

        Parameters
        ----------
        y : np.ndarray
            A k x 1 ndarray y representing the current measurement

        """
        self.prior_to_filtered(y)
        self.filtered_to_forecast()

    def stationary_values(self):
        """
        Computes the limit of :math:`Sigma_t` as :math:`t \to \infty` by
        solving the associated Riccati equation.  Computation is via the
        doubling algorithm (see the documentation in
        `matrix_eqn.solve_discrete_riccati`).

        Returns
        -------
        Sigma_infinity : array_like or scalar(float)
            The infinite limit of Sigma_t
        K_infinity : array_like or scalar(float)
            The stationary Kalman gain.

        """
        # === simplify notation === #
        A, Q, G, R = self.A, self.Q, self.G, self.R
        # === solve Riccati equation, obtain Kalman gain === #
        Sigma_infinity = solve_discrete_riccati(A.T, G.T, Q, R)
        temp1 = dot(dot(A, Sigma_infinity), G.T)
        temp2 = inv(dot(G, dot(Sigma_infinity, G.T)) + R)
        K_infinity = dot(temp1, temp2)

        return Sigma_infinity, K_infinity
