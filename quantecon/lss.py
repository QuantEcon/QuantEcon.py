"""
Origin: QE by Thomas J. Sargent and John Stachurski
Filename: lss.py
LastModified: 30/01/2014

Computes quantities related to the linear state space model

    x_{t+1} = A x_t + C w_{t+1}
        y_t = G x_t

The shocks {w_t} are iid and N(0, I)
"""

import numpy as np
from numpy import dot
from numpy.random import multivariate_normal
from scipy.linalg import eig, solve, solve_discrete_lyapunov

class LSS:

    def __init__(self, A, C, G, mu_0=None, Sigma_0=None):
        """
        Provides initial parameters describing the state space model

            x_{t+1} = A x_t + C w_{t+1}
                y_t = G x_t 

        where {w_t} are iid and N(0, I).  If the initial conditions mu_0 and
        Sigma_0 for x_0 ~ N(mu_0, Sigma_0) are not supplied, both are set to
        zero. When Sigma_0=0, the draw of x_0 is exactly mu_0.
        
        Parameters
        ============
        
        All arguments should be scalars or array_like

            * A is n x n
            * C is n x m
            * G is k x n
            * mu_0 is n x 1
            * Sigma_0 is n x n, positive definite and symmetric

        """
        self.A, self.G, self.C = map(self.convert, (A, G, C))
        self.k, self.n = self.G.shape
        self.m = self.C.shape[1]
        # == Default initial conditions == #
        if mu_0 == None:
            self.mu_0 = np.zeros((self.n, 1))
        else:
            self.mu_0 = np.asarray(mu_0)
        if Sigma_0 == None:
            self.Sigma_0 = np.zeros((self.n, self.n))
        else:
            self.Sigma_0 = Sigma_0

    def convert(self, x): 
        """
        Convert array_like objects (lists of lists, floats, etc.) into well
        formed 2D NumPy arrays
        """
        return np.atleast_2d(np.asarray(x, dtype='float32'))

    def simulate(self, ts_length=100):
        """
        Simulate a time series of length ts_length, first drawing 
        
            x_0 ~ N(mu_0, Sigma_0)


        Returns
        ========
        x : numpy.ndarray
            An n x ts_length array, where the t-th column is x_t

        y : numpy.ndarray
            A k x ts_length array, where the t-th column is y_t

        """
        x = np.empty((self.n, ts_length))
        x[:,0] = multivariate_normal(self.mu_0.flatten(), self.Sigma_0)
        w = np.random.randn(self.m, ts_length-1)
        for t in range(ts_length-1):
            x[:, t+1] = self.A.dot(x[:, t]) + self.C.dot(w[:, t])
        y = self.G.dot(x)
        return x, y

    def replicate(self, T=10, num_reps=100):
        """
        Simulate num_reps observations of x_T and y_T given 
        x_0 ~ N(mu_0, Sigma_0).

        Returns
        ========
        x : numpy.ndarray
            An n x num_reps array, where the j-th column is the j_th
            observation of x_T

        y : numpy.ndarray
            A k x num_reps array, where the j-th column is the j_th
            observation of y_T
        """
        x = np.empty((self.n, num_reps))
        for j in range(num_reps):
            x_T, _ = self.simulate(ts_length=T+1)
            x[:, j] = x_T[:, -1]
        y = self.G.dot(x)
        return x, y

    def moment_sequence(self):
        """
        Create a generator to calculate the population mean and
        variance-convariance matrix for both x_t and y_t, starting at the
        initial condition (self.mu_0, self.Sigma_0).  

        Returns
        ========

        A generator, such that each iteration produces the moments of x and y,
        updated one unit of time.  The moments are returned as a 4-tuple with
        the following interpretation:

        mu_x : numpy.ndarray
            An n x 1 array representing the population mean of x_t

        mu_y : numpy.ndarray
            A  k x 1 array representing the population mean of y_t

        Sigma_x : numpy.ndarray
            An n x n array representing the variance-covariance matrix of x_t

        Sigma_y : numpy.ndarray
            A k x k array representing the variance-covariance matrix of y_t

        """
        # == Simplify names == #
        A, C, G = self.A, self.C, self.G
        # == Initial moments == #
        mu_x, Sigma_x = self.mu_0, self.Sigma_0
        while 1:
            mu_y, Sigma_y = G.dot(mu_x), G.dot(Sigma_x).dot(G.T)
            yield mu_x, mu_y, Sigma_x, Sigma_y
            # == Update moments of x == #
            mu_x = A.dot(mu_x)
            Sigma_x = A.dot(Sigma_x).dot(A.T) + C.dot(C.T)

    def stationary_distributions(self, max_iter=200, tol=1e-5):
        """
        Compute the moments of the stationary distributions of x_t and y_t if
        possible.  Computation is by iteration, starting from the initial
        conditions self.mu_0 and self.Sigma_0

        Returns
        ========
        mu_x_star : numpy.ndarray
            An n x 1 array representing the stationary mean of x_t

        mu_y_star : numpy.ndarray
            An k x 1 array representing the stationary mean of y_t

        Sigma_x_star : numpy.ndarray
            An n x n array representing the stationary var-cov matrix of x_t

        Sigma_y_star : numpy.ndarray
            An k x k array representing the stationary var-cov matrix of y_t

        """
        # == Initialize iteration == #
        m = self.moment_sequence()
        mu_x, mu_y, Sigma_x, Sigma_y = m.next()
        i = 0
        error = tol + 1
        # == Loop until convergence or failuer == #
        while error > tol:

            if i > max_iter:
                fail_message = 'Convergence failed after {} iterations'
                raise ValueError(fail_message.format(max_iter))

            else:
                i += 1
                mu_x1, mu_y1, Sigma_x1, Sigma_y1 = m.next()
                error_mu = np.max(np.abs(mu_x1 - mu_x))
                error_Sigma = np.max(np.abs(Sigma_x1 - Sigma_x))
                error = max(error_mu, error_Sigma)
                mu_x, Sigma_x = mu_x1, Sigma_x1

        # == Prepare return values == #
        mu_x_star, Sigma_x_star = mu_x, Sigma_x
        mu_y_star, Sigma_y_star = mu_y1, Sigma_y1
        return mu_x_star, mu_y_star, Sigma_x_star, Sigma_y_star


    def geometric_sums(self, beta, x_t):
        """
        Forecast the geometric sums

            S_x := E [sum_{j=0}^{\infty} beta^j x_{t+j} | x_t ]

            S_y := E [sum_{j=0}^{\infty} beta^j y_{t+j} | x_t ]

        Parameters
        ===========
        beta : float
            Discount factor, in [0, 1)

        beta : array_like
            The term x_t for conditioning

        Returns
        ========
        S_x : numpy.ndarray
            Geometric sum as defined above

        S_y : numpy.ndarray
            Geometric sum as defined above

        """
        I = np.identity(self.n)
        S_x = solve(I - beta * self.A, x_t)
        S_y = self.G.dot(S_x)
        return S_x, S_y

