"""
Origin: QE by John Stachurski and Thomas J. Sargent
Filename: robustlq.py
Authors: Chase Coleman, Spencer Lyon, Thomas Sargent, John Stachurski 
LastModified: 28/01/2014

Solves robust LQ control problems.
"""

from __future__ import division  # Remove for Python 3.sx
import numpy as np
from lqcontrol import LQ
from quadsums import var_quadratic_sum
from numpy import dot, log, sqrt, identity, hstack, vstack, trace
from scipy.linalg import solve, inv, det, solve_discrete_lyapunov

class RBLQ:
    """
    Provides methods for analysing infinite horizon robust LQ control 
    problems of the form

        min_{u_t}  sum_t beta^t {x_t' R x_t + u'_t Q u_t }

    subject to
        
        x_{t+1} = A x_t + B u_t + C w_{t+1}

    and with model misspecification parameter theta.
    """

    def __init__(self, Q, R, A, B, C, beta, theta):
        """
        Sets up the robust control problem.

        Parameters
        ==========

        Q, R : array_like, dtype = float
            The matrices R and Q from the objective function

        A, B, C : array_like, dtype = float
            The matrices A, B, and C from the state space system

        beta, theta : scalar, float
            The discount and robustness factors in the robust control problem

        We assume that
        
            * R is n x n, symmetric and nonnegative definite
            * Q is k x k, symmetric and positive definite
            * A is n x n
            * B is n x k
            * C is n x j
            
        """
        # == Make sure all matrices can be treated as 2D arrays == #
        A, B, C, Q, R = map(np.atleast_2d, (A, B, C, Q, R))
        self.A, self.B, self.C, self.Q, self.R = A, B, C, Q, R
        # == Record dimensions == #
        self.k = self.Q.shape[0]
        self.n = self.R.shape[0]
        self.j = self.C.shape[1]
        # == Remaining parameters == #
        self.beta, self.theta = beta, theta

    def d_operator(self, P):
        """
        The D operator, mapping P into 
        
            D(P) := P + PC(theta I - C'PC)^{-1} C'P.

        Parameters
        ==========
        P : array_like
            A self.n x self.n array

        """
        C, theta = self.C, self.theta
        I = np.identity(self.j)
        S1 = dot(P, C)
        S2 = dot(C.T, S1)
        return P + dot(S1, solve(theta * I - S2, S1.T)) 

    def b_operator(self, P):
        """
        The B operator, mapping P into 
        
            B(P) := R - beta^2 A'PB (Q + beta B'PB)^{-1} B'PA + beta A'PA

        and also returning

            F := (Q + beta B'PB)^{-1} beta B'PA

        Parameters
        ==========
        P : array_like
            An self.n x self.n array

        """
        A, B, Q, R, beta = self.A, self.B, self.Q, self.R, self.beta
        S1 = Q + beta * dot(B.T, dot(P, B))   
        S2 = beta * dot(B.T, dot(P, A))
        S3 = beta * dot(A.T, dot(P, A))
        F = solve(S1, S2)  
        new_P = R - dot(S2.T, solve(S1, S2)) + S3  
        return F, new_P

    def robust_rule(self):
        """
        This method solves the robust control problem by tricking it into a
        stacked LQ problem, as described in chapter 2 of Hansen-Sargent's text
        "Robustness."  The optimal control with observed state is

            u_t = - F x_t

        And the value function is -x'Px

        Returns
        =======
        F : array_like, dtype = float
            The optimal control matrix from above above

        P : array_like, dtype = float
            The psoitive semi-definite matrix defining the value function

        K : array_like, dtype = float
            the worst-case shock matrix K, where :math:`w_{t+1} = K x_t` is
            the worst case shock

        """
        # == Simplify names == #
        A, B, C, Q, R = self.A, self.B, self.C, self.Q, self.R
        beta, theta = self.beta, self.theta
        k, j = self.k, self.j
        # == Set up LQ version == #
        I = identity(j)
        Z = np.zeros((k, j))
        Ba = hstack([B, C])
        Qa = vstack([hstack([Q, Z]), hstack([Z.T, -beta*I*theta])])
        lq = LQ(Qa, R, A, Ba, beta=beta)
        # == Solve and convert back to robust problem == #
        P, f, d = lq.stationary_values()
        F = f[:k, :]
        K = -f[k:f.shape[0], :]
        return F, K, P

    def robust_rule_simple(self, P_init=None, max_iter=80, tol=1e-8):
        """
        A simple algorithm for computing the robust policy F and the
        corresponding value function P, based around straightforward iteration
        with the robust Bellman operator.  This function is easier to
        understand but one or two orders of magnitude slower than
        self.robust_rule().  For more information see the docstring of that
        method.
        """
        # == Simplify names == #
        A, B, C, Q, R = self.A, self.B, self.C, self.Q, self.R
        beta, theta = self.beta, self.theta
        # == Set up loop == #
        P = np.zeros((self.n, self.n)) if not P_init else P_init
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

    def F_to_K(self, F):
        """
        Compute agent 2's best cost-minimizing response K, given F.

        Parameters
        ==========
        F : array_like
            A self.k x self.n array

        Returns
        =======
        K : array_like, dtype = float
        P : array_like, dtype = float

        """
        Q2 = self.beta * self.theta
        R2 = - self.R - dot(F.T, dot(self.Q, F))
        A2 = self.A - dot(self.B, F)
        B2 = self.C
        lq = LQ(Q2, R2, A2, B2, beta=self.beta)
        P, neg_K, d = lq.stationary_values()
        return - neg_K, P

    def K_to_F(self, K):
        """
        Compute agent 1's best value-maximizing response F, given K.

        Parameters
        ==========
        K : array_like
            A self.j x self.n array

        Returns
        =======
        F : array_like, dtype = float
        P : array_like, dtype = float

        """
        A1 = self.A + dot(self.C, K)
        B1 = self.B
        Q1 = self.Q
        R1 = self.R - self.beta * self.theta * dot(K.T, K)
        lq = LQ(Q1, R1, A1, B1, beta=self.beta)
        P, F, d = lq.stationary_values()
        return F, P

    def compute_deterministic_entropy(self, F, K, x0):
        """
        Given K and F, compute the value of deterministic entropy, which is 
        sum_t beta^t x_t' K'K x_t with x_{t+1} = (A - BF + CK) x_t.
        """
        H0 = dot(K.T, K)
        C0 = np.zeros((self.n, 1))
        A0 = self.A - dot(self.B, F) + dot(self.C, K)
        e = var_quadratic_sum(A0, C0, H0, self.beta, x0)
        return e

    def evaluate_F(self, F):
        """
        Given a fixed policy F, with the interpretation u = -F x, this
        function computes the matrix P_F and constant d_F associated with
        discounted cost J_F(x) = x' P_F x + d_F. 

        Parameters
        ==========
        F : array_like
            A self.k x self.n array

        Returns
        =======
        P_F : array_like, dtype = float
            Matrix for discounted cost

        d_F : scalar
            Constant for discounted cost

        K_F : array_like, dtype = float
            Worst case policy
            
        O_F : array_like, dtype = float
            Matrix for discounted entropy
            
        o_F : scalar
            Constant for discounted entropy

        
        """
        # == Simplify names == #
        Q, R, A, B, C = self.Q, self.R, self.A, self.B, self.C
        beta, theta = self.beta, self.theta
        # == Solve for policies and costs using agent 2's problem == #
        K_F, neg_P_F = self.F_to_K(F)
        P_F = - neg_P_F
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

