"""
Solves robust LQ control problems.

"""
from textwrap import dedent
import numpy as np
from .lqcontrol import LQ
from .quadsums import var_quadratic_sum
from numpy import dot, log, sqrt, identity, hstack, vstack, trace
from scipy.linalg import solve, inv, det
from .matrix_eqn import solve_discrete_lyapunov


class RBLQ:
    r"""
    Provides methods for analysing infinite horizon robust LQ control
    problems of the form

    .. math::

        \min_{u_t}  \sum_t \beta^t {x_t' R x_t + u_t' Q u_t }

    subject to

    .. math::

        x_{t+1} = A x_t + B u_t + C w_{t+1}

    and with model misspecification parameter theta.

    Parameters
    ----------
    Q : array_like(float, ndim=2)
        The cost(payoff) matrix for the controls.  See above for more.
        Q should be k x k and symmetric and positive definite
    R : array_like(float, ndim=2)
        The cost(payoff) matrix for the state.  See above for more. R
        should be n x n and symmetric and non-negative definite
    A : array_like(float, ndim=2)
        The matrix that corresponds with the state in the state space
        system.  A should be n x n
    B : array_like(float, ndim=2)
        The matrix that corresponds with the control in the state space
        system.  B should be n x k
    C : array_like(float, ndim=2)
        The matrix that corresponds with the random process in the
        state space system.  C should be n x j
    beta : scalar(float)
        The discount factor in the robust control problem
    theta : scalar(float)
        The robustness factor in the robust control problem

    Attributes
    ----------
    Q, R, A, B, C, beta, theta : see Parameters
    k, n, j : scalar(int)
        The dimensions of the matrices

    """

    def __init__(self, Q, R, A, B, C, beta, theta):

        # == Make sure all matrices can be treated as 2D arrays == #
        A, B, C, Q, R = list(map(np.atleast_2d, (A, B, C, Q, R)))
        self.A, self.B, self.C, self.Q, self.R = A, B, C, Q, R
        # == Record dimensions == #
        self.k = self.Q.shape[0]
        self.n = self.R.shape[0]
        self.j = self.C.shape[1]
        # == Remaining parameters == #
        self.beta, self.theta = beta, theta
        # == Check for case of no control (pure forecasting problem) == #
        self.pure_forecasting = True if not Q.any() and not B.any() else False

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        m = """\
        Robust linear quadratic control system
          - beta (discount parameter)   : {b}
          - theta (robustness factor)   : {th}
          - n (number of state variables)   : {n}
          - k (number of control variables) : {k}
          - j (number of shocks)            : {j}
        """
        return dedent(m.format(b=self.beta, n=self.n, k=self.k, j=self.j,
                               th=self.theta))

    def d_operator(self, P):
        r"""
        The D operator, mapping P into

        .. math::

            D(P) := P + PC(\theta I - C'PC)^{-1} C'P.

        Parameters
        ----------
        P : array_like(float, ndim=2)
            A matrix that should be n x n

        Returns
        -------
        dP : array_like(float, ndim=2)
            The matrix P after applying the D operator

        """
        C, theta = self.C, self.theta
        I = np.identity(self.j)
        S1 = dot(P, C)
        S2 = dot(C.T, S1)

        dP = P + dot(S1, solve(theta * I - S2, S1.T))

        return dP

    def b_operator(self, P):
        r"""
        The B operator, mapping P into

        .. math::

            B(P) := R - \beta^2 A'PB(Q + \beta B'PB)^{-1}B'PA + \beta A'PA

        and also returning

        .. math::

            F := (Q + \beta B'PB)^{-1} \beta B'PA

        Parameters
        ----------
        P : array_like(float, ndim=2)
            A matrix that should be n x n

        Returns
        -------
        F : array_like(float, ndim=2)
            The F matrix as defined above
        new_p : array_like(float, ndim=2)
            The matrix P after applying the B operator

        """
        A, B, Q, R, beta = self.A, self.B, self.Q, self.R, self.beta
        S1 = Q + beta * dot(B.T, dot(P, B))
        S2 = beta * dot(B.T, dot(P, A))
        S3 = beta * dot(A.T, dot(P, A))
        F = solve(S1, S2) if not self.pure_forecasting else np.zeros(
            (self.k, self.n))
        new_P = R - dot(S2.T, F) + S3

        return F, new_P

    def robust_rule(self, method='doubling'):
        """
        This method solves the robust control problem by tricking it
        into a stacked LQ problem, as described in chapter 2 of Hansen-
        Sargent's text "Robustness."  The optimal control with observed
        state is

        .. math::

            u_t = - F x_t

        And the value function is :math:`-x'Px`

        Parameters
        ----------
        method : str, optional(default='doubling')
            Solution method used in solving the associated Riccati
            equation, str in {'doubling', 'qz'}.
            
        Returns
        -------
        F : array_like(float, ndim=2)
            The optimal control matrix from above
        P : array_like(float, ndim=2)
            The positive semi-definite matrix defining the value
            function
        K : array_like(float, ndim=2)
            the worst-case shock matrix K, where
            :math:`w_{t+1} = K x_t` is the worst case shock

        """
        # == Simplify names == #
        A, B, C, Q, R = self.A, self.B, self.C, self.Q, self.R
        beta, theta = self.beta, self.theta
        k, j = self.k, self.j
        # == Set up LQ version == #
        I = identity(j)
        Z = np.zeros((k, j))

        if self.pure_forecasting:
            lq = LQ(-beta*I*theta, R, A, C, beta=beta)

            # == Solve and convert back to robust problem == #
            P, f, d = lq.stationary_values(method=method)
            F = np.zeros((self.k, self.n))
            K = -f[:k, :]

        else:
            Ba = hstack([B, C])
            Qa = vstack([hstack([Q, Z]), hstack([Z.T, -beta*I*theta])])
            lq = LQ(Qa, R, A, Ba, beta=beta)

            # == Solve and convert back to robust problem == #
            P, f, d = lq.stationary_values(method=method)
            F = f[:k, :]
            K = -f[k:f.shape[0], :]

        return F, K, P

    def robust_rule_simple(self, P_init=None, max_iter=80, tol=1e-8):
        """
        A simple algorithm for computing the robust policy F and the
        corresponding value function P, based around straightforward
        iteration with the robust Bellman operator.  This function is
        easier to understand but one or two orders of magnitude slower
        than self.robust_rule().  For more information see the docstring
        of that method.

        Parameters
        ----------
        P_init : array_like(float, ndim=2), optional(default=None)
            The initial guess for the value function matrix.  It will
            be a matrix of zeros if no guess is given
        max_iter : scalar(int), optional(default=80)
            The maximum number of iterations that are allowed
        tol : scalar(float), optional(default=1e-8)
            The tolerance for convergence

        Returns
        -------
        F : array_like(float, ndim=2)
            The optimal control matrix from above
        P : array_like(float, ndim=2)
            The positive semi-definite matrix defining the value
            function
        K : array_like(float, ndim=2)
            the worst-case shock matrix K, where
            :math:`w_{t+1} = K x_t` is the worst case shock

        """
        # == Simplify names == #
        A, B, C, Q, R = self.A, self.B, self.C, self.Q, self.R
        beta, theta = self.beta, self.theta
        # == Set up loop == #
        P = np.zeros((self.n, self.n)) if P_init is None else P_init
        iterate, e = 0, tol + 1
        while iterate < max_iter and e > tol:
            F, new_P = self.b_operator(self.d_operator(P))
            e = np.sqrt(np.sum((new_P - P)**2))
            iterate += 1
            P = new_P
        I = np.identity(self.j)
        S1 = P.dot(C)
        S2 = C.T.dot(S1)
        K = inv(theta * I - S2).dot(S1.T).dot(A - B.dot(F))

        return F, K, P

    def F_to_K(self, F, method='doubling'):
        """
        Compute agent 2's best cost-minimizing response K, given F.

        Parameters
        ----------
        F : array_like(float, ndim=2)
            A k x n array
        method : str, optional(default='doubling')
            Solution method used in solving the associated Riccati
            equation, str in {'doubling', 'qz'}.

        Returns
        -------
        K : array_like(float, ndim=2)
            Agent's best cost minimizing response for a given F
        P : array_like(float, ndim=2)
            The value function for a given F

        """
        Q2 = self.beta * self.theta
        R2 = - self.R - dot(F.T, dot(self.Q, F))
        A2 = self.A - dot(self.B, F)
        B2 = self.C
        lq = LQ(Q2, R2, A2, B2, beta=self.beta)
        neg_P, neg_K, d = lq.stationary_values(method=method)

        return -neg_K, -neg_P

    def K_to_F(self, K, method='doubling'):
        """
        Compute agent 1's best value-maximizing response F, given K.

        Parameters
        ----------
        K : array_like(float, ndim=2)
            A j x n array
        method : str, optional(default='doubling')
            Solution method used in solving the associated Riccati
            equation, str in {'doubling', 'qz'}.

        Returns
        -------
        F : array_like(float, ndim=2)
            The policy function for a given K
        P : array_like(float, ndim=2)
            The value function for a given K

        """
        A1 = self.A + dot(self.C, K)
        B1 = self.B
        Q1 = self.Q
        R1 = self.R - self.beta * self.theta * dot(K.T, K)
        lq = LQ(Q1, R1, A1, B1, beta=self.beta)
        P, F, d = lq.stationary_values(method=method)

        return F, P

    def compute_deterministic_entropy(self, F, K, x0):
        r"""

        Given K and F, compute the value of deterministic entropy, which
        is

        .. math::

            \sum_t \beta^t x_t' K'K x_t`

        with

        .. math::

            x_{t+1} = (A - BF + CK) x_t

        Parameters
        ----------
        F : array_like(float, ndim=2)
            The policy function, a k x n array
        K : array_like(float, ndim=2)
            The worst case matrix, a j x n array
        x0 : array_like(float, ndim=1)
            The initial condition for state

        Returns
        -------
        e : scalar(int)
            The deterministic entropy

        """
        H0 = dot(K.T, K)
        C0 = np.zeros((self.n, 1))
        A0 = self.A - dot(self.B, F) + dot(self.C, K)
        e = var_quadratic_sum(A0, C0, H0, self.beta, x0)

        return e

    def evaluate_F(self, F):
        """
        Given a fixed policy F, with the interpretation :math:`u = -F x`, this
        function computes the matrix :math:`P_F` and constant :math:`d_F`
        associated with discounted cost :math:`J_F(x) = x' P_F x + d_F`

        Parameters
        ----------
        F : array_like(float, ndim=2)
            The policy function, a k x n array

        Returns
        -------
        P_F : array_like(float, ndim=2)
            Matrix for discounted cost
        d_F : scalar(float)
            Constant for discounted cost
        K_F : array_like(float, ndim=2)
            Worst case policy
        O_F : array_like(float, ndim=2)
            Matrix for discounted entropy
        o_F : scalar(float)
            Constant for discounted entropy

        """
        # == Simplify names == #
        Q, R, A, B, C = self.Q, self.R, self.A, self.B, self.C
        beta, theta = self.beta, self.theta

        # == Solve for policies and costs using agent 2's problem == #
        K_F, P_F = self.F_to_K(F)
        I = np.identity(self.j)
        H = inv(I - C.T.dot(P_F.dot(C)) / theta)
        d_F = log(det(H))

        # == Compute O_F and o_F == #
        sig = -1.0 / theta
        AO = sqrt(beta) * (A - dot(B, F) + dot(C, K_F))
        O_F = solve_discrete_lyapunov(AO.T, beta * dot(K_F.T, K_F))
        ho = (trace(H - 1) - d_F) / 2.0
        tr = trace(dot(O_F, C.dot(H.dot(C.T))))
        o_F = (ho + beta * tr) / (1 - beta)

        return K_F, P_F, d_F, O_F, o_F
