"""
Filename: lqcontrol.py
Authors: Thomas J. Sargent, John Stachurski

Provides a class called LQ for solving linear quadratic control problems.
"""

import numpy as np
from numpy import dot
from scipy.linalg import solve
import riccati

class LQ:
    """
    This class is for analyzing linear quadratic optimal control problems of
    either the infinite horizon form

        min E sum_{t=0}^{infty} beta^t r(x_t, u_t)

    with

        r(x_t, u_t) := x_t' R x_t + u_t' Q u_t

    or the finite horizon form

    min E sum_{t=0}^{T-1} beta^t r(x_t, u_t) + x_T' R_f x_T

    Both are minimized subject to the law of motion

        x_{t+1} = A x_t + B u_t + C w_{t+1}

    Here x is n x 1, u is k x 1, w is j x 1 and the matrices are conformable
    for these dimensions.  The sequence {w_t} is assumed to be white noise,
    with zero mean and E w_t w_t' = I, the j x j identity.

    If C is not supplied as a parameter, the model is assumed to be
    deterministic (and C is set to a zero matrix of appropriate dimension).

    For this model, the time t value (i.e., cost-to-go) function V_t takes the
    form

        x' P_T x + d_T

    and the optimal policy is of the form u_T = -F_T x_T.  In the infinite
    horizon case, V, P, d and F are all stationary.
    """

    def __init__(self, Q, R, A, B, C=None, beta=1, T=None, Rf=None):
        """
        Provides parameters describing the LQ model

        Parameters
        ============

            * R and Rf are n x n, symmetric and nonnegative definite
            * Q is k x k, symmetric and positive definite
            * A is n x n
            * B is n x k
            * C is n x j, or None for a deterministic model
            * beta is a scalar in (0, 1] and T is an int

        All arguments should be scalars or NumPy ndarrays.

        Here T is the time horizon. If T is not supplied, then the LQ problem
        is assumed to be infinite horizon.  If T is supplied, then the
        terminal reward matrix Rf should also be specified.  For
        interpretation of the other parameters, see the docstring of the LQ
        class.

        We also initialize the pair (P, d) that represents the value function
        via V(x) = x' P x + d, and the policy function matrix F.
        """
        # == Make sure all matrices can be treated as 2D arrays == #
        converter = lambda X: np.atleast_2d(np.asarray(X, dtype='float32'))
        self.A, self.B, self.Q, self.R = map(converter, (A, B, Q, R))
        # == Record dimensions == #
        self.k, self.n = self.Q.shape[0], self.R.shape[0]

        self.beta = beta

        if C == None:
            # == If C not given, then model is deterministic. Set C=0. == #
            self.j = 1
            self.C = np.zeros((self.n, self.j))
        else:
            self.C = converter(C)
            self.j = self.C.shape[1]

        if T:
            # == Model is finite horizon == #
            self.T = T
            self.Rf = np.asarray(Rf, dtype='float32')
            self.P = self.Rf
            self.d = 0
        else:
            self.P = None
            self.d = None
            self.T = None

        self.F = None

    def update_values(self):
        """
        This method is for updating in the finite horizon case.  It shifts the
        current value function

            V_t(x) = x' P_t x + d_t

        and the optimal policy F_t one step *back* in time, replacing the pair
        P_t and d_t with P_{t-1} and d_{t-1}, and F_t with F_{t-1}
        """
        # === Simplify notation === #
        Q, R, A, B, C = self.Q, self.R, self.A, self.B, self.C
        P, d = self.P, self.d
        # == Some useful matrices == #
        S1 = Q + self.beta * dot(B.T, dot(P, B))
        S2 = self.beta * dot(B.T, dot(P, A))
        S3 = self.beta * dot(A.T, dot(P, A))
        # == Compute F as (Q + B'PB)^{-1} (beta B'PA) == #
        self.F = solve(S1, S2)
        # === Shift P back in time one step == #
        new_P = R - dot(S2.T, self.F) + S3
        # == Recalling that trace(AB) = trace(BA) == #
        new_d = self.beta * (d + np.trace(dot(P, dot(C, C.T))))
        # == Set new state == #
        self.P, self.d = new_P, new_d

    def stationary_values(self):
        """
        Computes the matrix P and scalar d that represent the value function

            V(x) = x' P x + d

        in the infinite horizon case.  Also computes the control matrix F from
        u = - Fx

        """
        # === simplify notation === #
        Q, R, A, B, C = self.Q, self.R, self.A, self.B, self.C
        # === solve Riccati equation, obtain P === #
        A0, B0 = np.sqrt(self.beta) * A, np.sqrt(self.beta) * B
        P = riccati.dare(A0, B0, Q, R)
        # == Compute F == #
        S1 = Q + self.beta * dot(B.T, dot(P, B))
        S2 = self.beta * dot(B.T, dot(P, A))
        F = solve(S1, S2)
        # == Compute d == #
        d = self.beta * np.trace(dot(P, dot(C, C.T))) / (1 - self.beta)
        # == Bind states and return values == #
        self.P, self.F, self.d = P, F, d
        return P, F, d

    def compute_sequence(self, x0, ts_length=None):
        """
        Compute and return the optimal state and control sequences x_0,...,
        x_T and u_0,..., u_T  under the assumption that {w_t} is iid and
        N(0, 1).

        Parameters
        ===========
        x0 : numpy.ndarray
            The initial state, a vector of length n

        ts_length : int
            Length of the simulation -- defaults to T in finite case

        Returns
        ========
        x_path : numpy.ndarray
            An n x T matrix, where the t-th column represents x_t

        u_path : numpy.ndarray
            A k x T matrix, where the t-th column represents u_t

        """
        # === Simplify notation === #
        Q, R, A, B, C = self.Q, self.R, self.A, self.B, self.C
        # == Preliminaries, finite horizon case == #
        if self.T:
            T = self.T if not ts_length else min(ts_length, self.T)
            self.P, self.d = self.Rf, 0
        # == Preliminaries, infinite horizon case == #
        else:
            T = ts_length if ts_length else 100
            self.stationary_values()
        # == Set up initial condition and arrays to store paths == #
        x0 = np.asarray(x0)
        x0 = x0.reshape(self.n, 1)  # Make sure x0 is a column vector
        x_path = np.empty((self.n, T+1))
        u_path = np.empty((self.k, T))
        w_path = dot(C, np.random.randn(self.j, T+1))
        # == Compute and record the sequence of policies == #
        policies = []
        for t in range(T):
            if self.T:  # Finite horizon case
                self.update_values()
            policies.append(self.F)
        # == Use policy sequence to generate states and controls == #
        F = policies.pop()
        x_path[:, 0] = x0.flatten()
        u_path[:, 0] = - dot(F, x0).flatten()
        for t in range(1, T):
            F = policies.pop()
            Ax, Bu = dot(A, x_path[:, t-1]), dot(B, u_path[:, t-1])
            x_path[:, t] =  Ax + Bu + w_path[:, t]
            u_path[:, t] = - dot(F, x_path[:, t])
        Ax, Bu = dot(A, x_path[:, T-1]), dot(B, u_path[:, T-1])
        x_path[:, T] =  Ax + Bu + w_path[:, T]
        return x_path, u_path, w_path
