"""
Filename: lqcontrol.py

Authors: Thomas J. Sargent, John Stachurski

Provides a class called LQ for solving linear quadratic control
problems.

"""
from textwrap import dedent
import numpy as np
from numpy import dot
from scipy.linalg import solve
from .matrix_eqn import solve_discrete_riccati
from .util import check_random_state


class LQ:
    r"""
    This class is for analyzing linear quadratic optimal control
    problems of either the infinite horizon form

    .. math::

        \min \mathbb{E}
        \Big[ \sum_{t=0}^{\infty} \beta^t r(x_t, u_t) \Big]

    with

    .. math::

         r(x_t, u_t) := x_t' R x_t + u_t' Q u_t + 2 u_t' N x_t

    or the finite horizon form

    .. math::

         \min \mathbb{E}
         \Big[
         \sum_{t=0}^{T-1} \beta^t r(x_t, u_t) + \beta^T x_T' R_f x_T
         \Big]

    Both are minimized subject to the law of motion

    .. math::

         x_{t+1} = A x_t + B u_t + C w_{t+1}

    Here :math:`x` is n x 1, :math:`u` is k x 1, :math:`w` is j x 1 and the
    matrices are conformable for these dimensions.  The sequence :math:`{w_t}`
    is assumed to be white noise, with zero mean and
    :math:`\mathbb{E} [ w_t' w_t ] = I`, the j x j identity.

    If :math:`C` is not supplied as a parameter, the model is assumed to be
    deterministic (and :math:`C` is set to a zero matrix of appropriate
    dimension).

    For this model, the time t value (i.e., cost-to-go) function :math:`V_t`
    takes the form

    .. math::

         x' P_T x + d_T

    and the optimal policy is of the form :math:`u_T = -F_T x_T`. In the
    infinite horizon case, :math:`V, P, d` and :math:`F` are all stationary.

    Parameters
    ----------
    Q : array_like(float)
        Q is the payoff (or cost) matrix that corresponds with the
        control variable u and is k x k. Should be symmetric and
        non-negative definite
    R : array_like(float)
        R is the payoff (or cost) matrix that corresponds with the
        state variable x and is n x n. Should be symetric and
        non-negative definite
    A : array_like(float)
        A is part of the state transition as described above. It should
        be n x n
    B : array_like(float)
        B is part of the state transition as described above. It should
        be n x k
    C : array_like(float), optional(default=None)
        C is part of the state transition as described above and
        corresponds to the random variable today.  If the model is
        deterministic then C should take default value of None
    N : array_like(float), optional(default=None)
        N is the cross product term in the payoff, as above. It should
        be k x n.
    beta : scalar(float), optional(default=1)
        beta is the discount parameter
    T : scalar(int), optional(default=None)
        T is the number of periods in a finite horizon problem.
    Rf : array_like(float), optional(default=None)
        Rf is the final (in a finite horizon model) payoff(or cost)
        matrix that corresponds with the control variable u and is n x
        n.  Should be symetric and non-negative definite

    Attributes
    ----------
    Q, R, N, A, B, C, beta, T, Rf : see Parameters
    P : array_like(float)
        P is part of the value function representation of
        :math:`V(x) = x'Px + d`
    d : array_like(float)
        d is part of the value function representation of
        :math:`V(x) = x'Px + d`
    F : array_like(float)
        F is the policy rule that determines the choice of control in
        each period.
    k, n, j : scalar(int)
        The dimensions of the matrices as presented above

    """

    def __init__(self, Q, R, A, B, C=None, N=None, beta=1, T=None, Rf=None):
        # == Make sure all matrices can be treated as 2D arrays == #
        converter = lambda X: np.atleast_2d(np.asarray(X, dtype='float'))
        self.A, self.B, self.Q, self.R, self.N = list(map(converter,
                                                          (A, B, Q, R, N)))
        # == Record dimensions == #
        self.k, self.n = self.Q.shape[0], self.R.shape[0]

        self.beta = beta

        if C is None:
            # == If C not given, then model is deterministic. Set C=0. == #
            self.j = 1
            self.C = np.zeros((self.n, self.j))
        else:
            self.C = converter(C)
            self.j = self.C.shape[1]

        if N is None:
            # == No cross product term in payoff. Set N=0. == #
            self.N = np.zeros((self.k, self.n))

        if T:
            # == Model is finite horizon == #
            self.T = T
            self.Rf = np.asarray(Rf, dtype='float')
            self.P = self.Rf
            self.d = 0
        else:
            self.P = None
            self.d = None
            self.T = None

        self.F = None

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        m = """\
        Linear Quadratic control system
          - beta (discount parameter)       : {b}
          - T (time horizon)                : {t}
          - n (number of state variables)   : {n}
          - k (number of control variables) : {k}
          - j (number of shocks)            : {j}
        """
        t = "infinite" if self.T is None else self.T
        return dedent(m.format(b=self.beta, n=self.n, k=self.k, j=self.j,
                               t=t))

    def update_values(self):
        """
        This method is for updating in the finite horizon case.  It
        shifts the current value function

        .. math::

             V_t(x) = x' P_t x + d_t

        and the optimal policy :math:`F_t` one step *back* in time,
        replacing the pair :math:`P_t` and :math:`d_t` with
        :math:`P_{t-1}` and :math:`d_{t-1}`, and :math:`F_t` with
        :math:`F_{t-1}`

        """
        # === Simplify notation === #
        Q, R, A, B, N, C = self.Q, self.R, self.A, self.B, self.N, self.C
        P, d = self.P, self.d
        # == Some useful matrices == #
        S1 = Q + self.beta * dot(B.T, dot(P, B))
        S2 = self.beta * dot(B.T, dot(P, A)) + N
        S3 = self.beta * dot(A.T, dot(P, A))
        # == Compute F as (Q + B'PB)^{-1} (beta B'PA + N) == #
        self.F = solve(S1, S2)
        # === Shift P back in time one step == #
        new_P = R - dot(S2.T, self.F) + S3
        # == Recalling that trace(AB) = trace(BA) == #
        new_d = self.beta * (d + np.trace(dot(P, dot(C, C.T))))
        # == Set new state == #
        self.P, self.d = new_P, new_d

    def stationary_values(self, method='doubling'):
        """
        Computes the matrix :math:`P` and scalar :math:`d` that represent
        the value function

        .. math::

             V(x) = x' P x + d

        in the infinite horizon case.  Also computes the control matrix
        :math:`F` from :math:`u = - Fx`. Computation is via the solution
        algorithm as specified by the `method` option (default to the
        doubling algorithm) (see the documentation in
        `matrix_eqn.solve_discrete_riccati`).

        Parameters
        ----------
        method : str, optional(default='doubling')
            Solution method used in solving the associated Riccati
            equation, str in {'doubling', 'qz'}.

        Returns
        -------
        P : array_like(float)
            P is part of the value function representation of
            :math:`V(x) = x'Px + d`
        F : array_like(float)
            F is the policy rule that determines the choice of control
            in each period.
        d : array_like(float)
            d is part of the value function representation of
            :math:`V(x) = x'Px + d`

        """
        # === simplify notation === #
        Q, R, A, B, N, C = self.Q, self.R, self.A, self.B, self.N, self.C

        # === solve Riccati equation, obtain P === #
        A0, B0 = np.sqrt(self.beta) * A, np.sqrt(self.beta) * B
        P = solve_discrete_riccati(A0, B0, R, Q, N, method=method)

        # == Compute F == #
        S1 = Q + self.beta * dot(B.T, dot(P, B))
        S2 = self.beta * dot(B.T, dot(P, A)) + N
        F = solve(S1, S2)

        # == Compute d == #
        d = self.beta * np.trace(dot(P, dot(C, C.T))) / (1 - self.beta)

        # == Bind states and return values == #
        self.P, self.F, self.d = P, F, d

        return P, F, d

    def compute_sequence(self, x0, ts_length=None, method='doubling',
                         random_state=None):
        """
        Compute and return the optimal state and control sequences
        :math:`x_0, ..., x_T` and :math:`u_0,..., u_T`  under the
        assumption that :math:`{w_t}` is iid and :math:`N(0, 1)`.

        Parameters
        ----------
        x0 : array_like(float)
            The initial state, a vector of length n

        ts_length : scalar(int)
            Length of the simulation -- defaults to T in finite case

        method : str, optional(default='doubling')
            Solution method used in solving the associated Riccati
            equation, str in {'doubling', 'qz'}. Only relevant when the
            `T` attribute is `None` (i.e., the horizon is infinite).

        random_state : int or np.random.RandomState, optional
            Random seed (integer) or np.random.RandomState instance to set
            the initial state of the random number generator for
            reproducibility. If None, a randomly initialized RandomState is
            used.

        Returns
        -------
        x_path : array_like(float)
            An n x T+1 matrix, where the t-th column represents :math:`x_t`

        u_path : array_like(float)
            A k x T matrix, where the t-th column represents :math:`u_t`

        w_path : array_like(float)
            A j x T+1 matrix, where the t-th column represent :math:`w_t`

        """

        # === Simplify notation === #
        A, B, C = self.A, self.B, self.C

        # == Preliminaries, finite horizon case == #
        if self.T:
            T = self.T if not ts_length else min(ts_length, self.T)
            self.P, self.d = self.Rf, 0

        # == Preliminaries, infinite horizon case == #
        else:
            T = ts_length if ts_length else 100
            self.stationary_values(method=method)

        # == Set up initial condition and arrays to store paths == #
        random_state = check_random_state(random_state)
        x0 = np.asarray(x0)
        x0 = x0.reshape(self.n, 1)  # Make sure x0 is a column vector
        x_path = np.empty((self.n, T+1))
        u_path = np.empty((self.k, T))
        w_path = random_state.randn(self.j, T+1)
        Cw_path = dot(C, w_path)

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
            x_path[:, t] = Ax + Bu + Cw_path[:, t]
            u_path[:, t] = - dot(F, x_path[:, t])
        Ax, Bu = dot(A, x_path[:, T-1]), dot(B, u_path[:, T-1])
        x_path[:, T] = Ax + Bu + Cw_path[:, T]

        return x_path, u_path, w_path
