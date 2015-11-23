"""
Filename: kalman.py
Reference: http://quant-econ.net/py/kalman.html

Implements the Kalman filter for a linear Gaussian state space model.

"""
from textwrap import dedent
import numpy as np
from numpy import dot
from scipy.linalg import inv
from quantecon.lss import LinearStateSpace
from quantecon.matrix_eqn import solve_discrete_riccati


class Kalman(object):
    r"""
    Implements the Kalman filter for the Gaussian state space model

        x_{t+1} = A x_t + C w_{t+1}
        y_t = G x_t + H v_t.

    Here x_t is the hidden state and y_t is the measurement. The shocks
    w_t and v_t are iid standard normals.  Below we use the notation

        Q := CC'
        R := HH'


    Parameters
    -----------
    ss : instance of LinearStateSpace
        An instance of the quantecon.lss.LinearStateSpace class
    x_hat : scalar(float) or array_like(float), optional(default=None)
        An n x 1 array representing the mean x_hat and covariance
        matrix Sigma of the prior/predictive density.  Set to zero if
        not supplied.
    Sigma : scalar(float) or array_like(float), optional(default=None)
        An n x n array representing the covariance matrix Sigma of
        the prior/predictive density.  Must be positive definite.
        Set to the identity if not supplied.

    Attributes
    ----------
    Sigma, x_hat : as above
    Sigma_infinity : array_like or scalar(float)
        The infinite limit of Sigma_t
    K_infinity : array_like or scalar(float)
        The stationary Kalman gain.


    References
    ----------

    http://quant-econ.net/py/kalman.html

    """

    def __init__(self, ss, x_hat=None, Sigma=None):
        self.ss = ss
        self.set_state(x_hat, Sigma)
        self.K_infinity = None
        self.Sigma_infinity = None

    def set_state(self, x_hat, Sigma):
        if Sigma is None:
            Sigma = np.identity(self.ss.n)
        else:
            self.Sigma = np.atleast_2d(Sigma)
        if x_hat is None:
            x_hat = np.zeros((self.ss.n, 1))
        else:
            self.x_hat = np.atleast_2d(x_hat)
            self.x_hat.shape = self.ss.n, 1

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        m = """\
        Kalman filter:
          - dimension of state space          : {n}
          - dimension of observation equation : {k}
        """
        return dedent(m.format(n=self.ss.n, k=self.ss.k))

    def whitener_lss(self):
        r"""
        This function takes the linear state space system
        that is an input to the Kalman class and it converts
        that system to the time-invariant whitener represenation
        given by

            \tilde{x}_{t+1}^* = \tilde{A} \tilde{x} + \tilde{C} v
            a = \tilde{G} \tilde{x}

        where

            \tilde{x}_t = [x+{t}, \hat{x}_{t}, v_{t}]

        and

            \tilde{A} = [A  0    0
                         KG A-KG KH
                         0  0    0]

            \tilde{C} = [C 0
                         0 0
                         0 I]

            \tilde{G} = [G -G H]

        with A, C, G, H coming from the linear state space system 
        that defines the Kalman instance


        Returns
        -------
        whitened_lss : LinearStateSpace
            This is the linear state space system that represents
            the whitened system
        """
        # Check for steady state Sigma and K
        if self.K_infinity is None:
            Sig, K = self.stationary_values()
            self.Sigma_infinity = Sig
            self.K_infinity = K
        else:
            K = self.K_infinity

        # Get the matrix sizes
        n, k, m, l = self.ss.n, self.ss.k, self.ss.m, self.ss.l
        A, C, G, H = self.ss.A, self.ss.C, self.ss.G, self.ss.H

        Atil = np.vstack([np.hstack([A, np.zeros((n, n)), np.zeros((n, l))]),
                          np.hstack([dot(K, G), A-dot(K, G), dot(K, H)]),
                          np.zeros((l, 2*n + l))])

        Ctil = np.vstack([np.hstack([C, np.zeros((n, l))]),
                          np.zeros((n, m+l)),
                          np.hstack([np.zeros((l, m)), np.eye(l)])])

        Gtil = np.hstack([G, -G, H])

        whitened_lss = LinearStateSpace(Atil, Ctil, Gtil)
        self.whitened_lss = whitened_lss

        return whitened_lss


    def prior_to_filtered(self, y):
        r"""
        Updates the moments (x_hat, Sigma) of the time t prior to the
        time t filtering distribution, using current measurement y_t.

        The updates are according to

            x_{hat}^F = x_{hat} + Sigma G' (G Sigma G' + R)^{-1}
                (y - G x_{hat})
            Sigma^F = Sigma - Sigma G' (G Sigma G' + R)^{-1} G
                Sigma

        Parameters
        ----------
        y : scalar or array_like(float)
            The current measurement

        """
        # === simplify notation === #
        G, H = self.ss.G, self.ss.H
        R = np.dot(H, H.T)

        # === and then update === #
        y = np.atleast_2d(y)
        y.shape = self.ss.k, 1
        E = dot(self.Sigma, G.T)
        F = dot(dot(G, self.Sigma), G.T) + R
        M = dot(E, inv(F))
        self.x_hat = self.x_hat + dot(M, (y - dot(G, self.x_hat)))
        self.Sigma = self.Sigma - dot(M, dot(G,  self.Sigma))

    def filtered_to_forecast(self):
        """
        Updates the moments of the time t filtering distribution to the
        moments of the predictive distribution, which becomes the time
        t+1 prior

        """
        # === simplify notation === #
        A, C = self.ss.A, self.ss.C
        Q = np.dot(C, C.T)

        # === and then update === #
        self.x_hat = dot(A, self.x_hat)
        self.Sigma = dot(A, dot(self.Sigma, A.T)) + Q

    def update(self, y):
        """
        Updates x_hat and Sigma given k x 1 ndarray y.  The full
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
        Computes the limit of Sigma_t as t  goes to infinity by
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
        A, C, G, H = self.ss.A, self.ss.C, self.ss.G, self.ss.H
        Q, R = np.dot(C, C.T), np.dot(H, H.T)

        # === solve Riccati equation, obtain Kalman gain === #
        Sigma_infinity = solve_discrete_riccati(A.T, G.T, Q, R)
        temp1 = dot(dot(A, Sigma_infinity), G.T)
        temp2 = inv(dot(G, dot(Sigma_infinity, G.T)) + R)
        K_infinity = dot(temp1, temp2)

        # == record as attributes and return == #
        self.Sigma_infinity, self.K_infinity = Sigma_infinity, K_infinity
        return Sigma_infinity, K_infinity

    def stationary_coefficients(self, j, coeff_type='ma'):
        """
        Wold representation moving average or VAR coefficients for the
        steady state Kalman filter.

        Parameters
        ----------
        j : int
            The lag length
        coeff_type : string, either 'ma' or 'var' (default='ma')
            The type of coefficent sequence to compute.  Either 'ma' for
            moving average or 'var' for VAR.
        """
        # == simplify notation == #
        A, G = self.ss.A, self.ss.G
        K_infinity = self.K_infinity
        # == make sure that K_infinity has actually been computed == #
        if K_infinity is None:
            S, K_infinity = self.stationary_values()
        # == compute and return coefficients == #
        coeffs = []
        i = 1
        if coeff_type == 'ma':
            coeffs.append(np.identity(self.ss.k))
            P_mat = A
            P = np.identity(self.ss.n)  # Create a copy
        elif coeff_type == 'var':
            coeffs.append(dot(G, K_infinity))
            P_mat = A - dot(K_infinity, G)
            P = np.copy(P_mat)  # Create a copy
        else:
            raise ValueError("Unknown coefficient type")
        while i <= j:
            coeffs.append(dot(dot(G, P), K_infinity))
            P = dot(P, P_mat)
            i += 1
        return coeffs

    def stationary_innovation_covar(self):
        # == simplify notation == #
        H, G = self.ss.H, self.ss.G
        R = np.dot(H, H.T)
        Sigma_infinity = self.Sigma_infinity

        # == make sure that Sigma_infinity has been computed == #
        if Sigma_infinity is None:
            Sigma_infinity, K = self.stationary_values()
        return dot(G, dot(Sigma_infinity, G.T)) + R
