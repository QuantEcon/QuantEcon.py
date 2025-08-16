"""
Provides a class called DLE to convert and solve dynamic linear economies
(as set out in Hansen & Sargent (2013)) as LQ problems.
"""

import numpy as np
from ._lqcontrol import LQ
from ._matrix_eqn import solve_discrete_lyapunov
from ._rank_nullspace import nullspace

class DLE(object):
    r"""
    This class is for analyzing dynamic linear economies, as set out in Hansen & Sargent (2013).
    The planner's problem is to choose \{c_t, s_t, i_t, h_t, k_t, g_t\}_{t=0}^\infty to maximize

        \max -(1/2) \mathbb{E} \sum_{t=0}^{\infty} \beta^t [(s_t - b_t).(s_t-b_t) + g_t.g_t]

    subject to the linear constraints

        \Phi_c c_t + \Phi_g g_t + \Phi_i i_t = \Gamma k_{t-1} + d_t
        k_t = \Delta_k k_{t-1} + \Theta_k i_t
        h_t = \Delta_h h_{t-1} + \Theta_h c_t
        s_t = \Lambda h_{t-1} + \Pi c_t

    and

        z_{t+1} = A_{22} z_t + C_2 w_{t+1}
        b_t = U_b z_t
        d_t = U_d z_t

    where h_{-1}, k_{-1}, and z_0 are given as initial conditions.

    Section 5.5 of HS2013 describes how to map these matrices into those of
    a LQ problem.

    HS2013 sort the matrices defining the problem into three groups:

    Information: A_{22}, C_2, U_b , and U_d characterize the motion of information
    sets and of taste and technology shocks

    Technology: \Phi_c, \Phi_g, \Phi_i, \Gamma, \Delta_k, and \Theta_k determine the
    technology for producing consumption goods

    Preferences: \Delta_h, \Theta_h, \Lambda, and \Pi determine the technology for
    producing consumption services from consumer goods. A scalar discount factor \beta
    determines the preference ordering over consumption services.

    Parameters
    ----------
    Information : tuple
        Information is a tuple containing the matrices A_{22}, C_2, U_b, and U_d
    Technology : tuple
        Technology is a tuple containing the matrices \Phi_c, \Phi_g, \Phi_i, \Gamma,
        \Delta_k, and \Theta_k
    Preferences : tuple
        Preferences is a tuple containing the scalar \beta and the
        matrices \Lambda, \Pi, \Delta_h, and \Theta_h

    """

    def __init__(self, information, technology, preferences):

        # === Unpack the tuples which define information, technology and preferences === #
        self.a22, self.c2, self.ub, self.ud = information
        self.phic, self.phig, self.phii, self.gamma, self.deltak, self.thetak = technology
        self.beta, self.llambda, self.pih, self.deltah, self.thetah = preferences

        # === Computation of the dimension of the structural parameter matrices === #
        self.nb, self.nh = self.llambda.shape
        self.nd, self.nc = self.phic.shape
        self.nz, self.nw = self.c2.shape
        _, self.ng = self.phig.shape
        self.nk, self.ni = self.thetak.shape

        # === Creation of various useful matrices === #
        uc = np.hstack((np.eye(self.nc), np.zeros((self.nc, self.ng))))
        ug = np.hstack((np.zeros((self.ng, self.nc)), np.eye(self.ng)))
        phiin = np.linalg.inv(np.hstack((self.phic, self.phig)))
        phiinc = uc @ phiin
        b11 = - self.thetah @ phiinc @ self.phii
        a1 = self.thetah @ phiinc @ self.gamma
        a12 = np.vstack((self.thetah @ phiinc @
            self.ud, np.zeros((self.nk, self.nz))))

        # === Creation of the A Matrix for the state transition of the LQ problem === #

        a11 = np.vstack((np.hstack((self.deltah, a1)), np.hstack(
            (np.zeros((self.nk, self.nh)), self.deltak))))
        self.A = np.vstack((np.hstack((a11, a12)), np.hstack(
            (np.zeros((self.nz, self.nk + self.nh)), self.a22))))

        # === Creation of the B Matrix for the state transition of the LQ problem === #

        b1 = np.vstack((b11, self.thetak))
        self.B = np.vstack((b1, np.zeros((self.nz, self.ni))))

        # === Creation of the C Matrix for the state transition of the LQ problem === #

        self.C = np.vstack((np.zeros((self.nk + self.nh, self.nw)), self.c2))

        # === Define R,W and Q for the payoff function of the LQ problem === #

        self.H = np.hstack((self.llambda, self.pih @ uc @ phiin @ self.gamma, self.pih @
            uc @ phiin @ self.ud - self.ub, -self.pih @ uc @ phiin @ self.phii))
        self.G = (ug @ phiin @
            np.hstack((np.zeros((self.nd, self.nh)), self.gamma, self.ud, -self.phii)))
        self.S = (self.G.T @ self.G + self.H.T @ self.H) / 2

        self.nx = self.nh + self.nk + self.nz
        self.n = self.ni + self.nh + self.nk + self.nz

        self.R = self.S[0:self.nx, 0:self.nx]
        self.W = self.S[self.nx:self.n, 0:self.nx]
        self.Q = self.S[self.nx:self.n, self.nx:self.n]

        # === Use quantecon's LQ code to solve our LQ problem === #

        lq = LQ(self.Q, self.R, self.A, self.B,
                self.C, N=self.W, beta=self.beta)

        self.P, self.F, self.d = lq.stationary_values()

        # === Construct output matrices for our economy using the solution to the LQ problem === #

        self.A0 = self.A - self.B @ self.F

        self.Sh = self.A0[0:self.nh, 0:self.nx]
        self.Sk = self.A0[self.nh:self.nh + self.nk, 0:self.nx]
        self.Sk1 = np.hstack((np.zeros((self.nk, self.nh)), np.eye(
            self.nk), np.zeros((self.nk, self.nz))))
        self.Si = -self.F
        self.Sd = np.hstack((np.zeros((self.nd, self.nh + self.nk)), self.ud))
        self.Sb = np.hstack((np.zeros((self.nb, self.nh + self.nk)), self.ub))
        self.Sc = uc @ phiin @ (-self.phii @ self.Si +
                                    self.gamma @ self.Sk1 + self.Sd)
        self.Sg = ug @ phiin @ (-self.phii @ self.Si +
                                    self.gamma @ self.Sk1 + self.Sd)
        self.Ss = self.llambda @ np.hstack((np.eye(self.nh), np.zeros(
            (self.nh, self.nk + self.nz)))) + self.pih @ self.Sc

        # ===  Calculate eigenvalues of A0 === #
        self.A110 = self.A0[0:self.nh + self.nk, 0:self.nh + self.nk]
        self.endo = np.linalg.eigvals(self.A110)
        self.exo = np.linalg.eigvals(self.a22)

        # === Construct matrices for Lagrange Multipliers === #

        self.Mk = -2 * self.beta.item() * (np.hstack((np.zeros((self.nk, self.nh)), np.eye(
            self.nk), np.zeros((self.nk, self.nz))))) @ self.P @ self.A0
        self.Mh = -2 * self.beta.item() * (np.hstack((np.eye(self.nh), np.zeros(
            (self.nh, self.nk)), np.zeros((self.nh, self.nz))))) @ self.P @ self.A0
        self.Ms = -(self.Sb - self.Ss)
        self.Md = -(np.linalg.inv(np.vstack((self.phic.T, self.phig.T))) @
            np.vstack((self.thetah.T @ self.Mh + self.pih.T @ self.Ms, -self.Sg)))
        self.Mc = -(self.thetah.T @ self.Mh + self.pih.T @ self.Ms)
        self.Mi = -(self.thetak.T @ self.Mk)

    def compute_steadystate(self, nnc=2):
        """
        Computes the non-stochastic steady-state of the economy.

        Parameters
        ----------
        nnc : array_like(float)
            nnc is the location of the constant in the state vector x_t

        """
        zx = np.eye(self.A0.shape[0])-self.A0
        self.zz = nullspace(zx)
        self.zz /= self.zz[nnc]
        self.css = self.Sc @ self.zz
        self.sss = self.Ss @ self.zz
        self.iss = self.Si @ self.zz
        self.dss = self.Sd @ self.zz
        self.bss = self.Sb @ self.zz
        self.kss = self.Sk @ self.zz
        self.hss = self.Sh @ self.zz

    def compute_sequence(self, x0, ts_length=None, Pay=None):
        """
        Simulate quantities and prices for the economy

        Parameters
        ----------
        x0 : array_like(float)
            The initial state

        ts_length : scalar(int)
            Length of the simulation

        Pay : array_like(float)
            Vector to price an asset whose payout is Pay*xt

        """
        lq = LQ(self.Q, self.R, self.A, self.B,
                self.C, N=self.W, beta=self.beta)
        xp, up, wp = lq.compute_sequence(x0, ts_length)
        self.h = self.Sh @ xp
        self.k = self.Sk @ xp
        self.i = self.Si @ xp
        self.b = self.Sb @ xp
        self.d = self.Sd @ xp
        self.c = self.Sc @ xp
        self.g = self.Sg @ xp
        self.s = self.Ss @ xp

        # === Value of J-period risk-free bonds === #
        # === See p.145: Equation (7.11.2) === #
        e1 = np.zeros((1, self.nc))
        e1[0, 0] = 1
        self.R1_Price = np.empty((ts_length + 1, 1))
        self.R2_Price = np.empty((ts_length + 1, 1))
        self.R5_Price = np.empty((ts_length + 1, 1))
        for i in range(ts_length + 1):
            self.R1_Price[i, 0] = self.beta * e1 @ self.Mc @ np.linalg.matrix_power(
                self.A0, 1) @ xp[:, i] / (e1 @ self.Mc @ xp[:, i])
            self.R2_Price[i, 0] = self.beta**2 * (e1 @ self.Mc @
                np.linalg.matrix_power(self.A0, 2) @ xp[:, i]) / (e1 @ self.Mc @ xp[:, i])
            self.R5_Price[i, 0] = self.beta**5 * (e1 @ self.Mc @
                np.linalg.matrix_power(self.A0, 5) @ xp[:, i]) / (e1 @ self.Mc @ xp[:, i])

        # === Gross rates of return on 1-period risk-free bonds === #
        self.R1_Gross = 1 / self.R1_Price

        # === Net rates of return on J-period risk-free bonds === #
        # === See p.148: log of gross rate of return, divided by j === #
        self.R1_Net = np.log(1 / self.R1_Price) / 1
        self.R2_Net = np.log(1 / self.R2_Price) / 2
        self.R5_Net = np.log(1 / self.R5_Price) / 5

        # === Value of asset whose payout vector is Pay*xt === #
        # See p.145: Equation (7.11.1)
        if isinstance(Pay, np.ndarray) == True:
            self.Za = Pay.T @ self.Mc
            self.Q = solve_discrete_lyapunov(
                self.A0.T * self.beta**0.5, self.Za)
            self.q = self.beta / (1 - self.beta) * \
                np.trace(self.C.T @ self.Q @ self.C)
            self.Pay_Price = np.empty((ts_length + 1, 1))
            self.Pay_Gross = np.empty((ts_length + 1, 1))
            self.Pay_Gross[0, 0] = np.nan
            for i in range(ts_length + 1):
                self.Pay_Price[i, 0] = (xp[:, i].T @ self.Q @
                    xp[:, i] + self.q) / (e1 @ self.Mc @ xp[:, i])
            for i in range(ts_length):
                self.Pay_Gross[i + 1, 0] = self.Pay_Price[i + 1,
                                                          0] / (self.Pay_Price[i, 0] - Pay @ xp[:, i])
        return

    def irf(self, ts_length=100, shock=None):
        """
        Create Impulse Response Functions

        Parameters
        ----------

        ts_length : scalar(int)
            Number of periods to calculate IRF

        Shock : array_like(float)
            Vector of shocks to calculate IRF to. Default is first element of w

        """

        if type(shock) != np.ndarray:
            # Default is to select first element of w
            shock = np.vstack((np.ones((1, 1)), np.zeros((self.nw - 1, 1))))

        self.c_irf = np.empty((ts_length, self.nc))
        self.s_irf = np.empty((ts_length, self.nb))
        self.i_irf = np.empty((ts_length, self.ni))
        self.k_irf = np.empty((ts_length, self.nk))
        self.h_irf = np.empty((ts_length, self.nh))
        self.g_irf = np.empty((ts_length, self.ng))
        self.d_irf = np.empty((ts_length, self.nd))
        self.b_irf = np.empty((ts_length, self.nb))

        for i in range(ts_length):
            self.c_irf[i, :] = (self.Sc @
                np.linalg.matrix_power(self.A0, i) @ self.C @ shock).T
            self.s_irf[i, :] = (self.Ss @
                np.linalg.matrix_power(self.A0, i) @ self.C @ shock).T
            self.i_irf[i, :] = (self.Si @
                np.linalg.matrix_power(self.A0, i) @ self.C @ shock).T
            self.k_irf[i, :] = (self.Sk @
                np.linalg.matrix_power(self.A0, i) @ self.C @ shock).T
            self.h_irf[i, :] = (self.Sh @
                np.linalg.matrix_power(self.A0, i) @ self.C @ shock).T
            self.g_irf[i, :] = (self.Sg @
                np.linalg.matrix_power(self.A0, i) @ self.C @ shock).T
            self.d_irf[i, :] = (self.Sd @
                np.linalg.matrix_power(self.A0, i) @ self.C @ shock).T
            self.b_irf[i, :] = (self.Sb @
                np.linalg.matrix_power(self.A0, i) @ self.C @ shock).T

        return

    def canonical(self):
        """
        Compute canonical preference representation
        Uses auxiliary problem of 9.4.2, with the preference shock process reintroduced
        Calculates pihat, llambdahat and ubhat for the equivalent canonical household technology
        """
        Ac1 = np.hstack((self.deltah, np.zeros((self.nh, self.nz))))
        Ac2 = np.hstack((np.zeros((self.nz, self.nh)), self.a22))
        Ac = np.vstack((Ac1, Ac2))
        Bc = np.vstack((self.thetah, np.zeros((self.nz, self.nc))))
        Rc1 = np.hstack((self.llambda.T @ self.llambda, -
                         self.llambda.T @ self.ub))
        Rc2 = np.hstack((-self.ub.T @ self.llambda, self.ub.T @ self.ub))
        Rc = np.vstack((Rc1, Rc2))
        Qc = self.pih.T @ self.pih
        Nc = np.hstack(
            (self.pih.T @ self.llambda, -self.pih.T @ self.ub))

        lq_aux = LQ(Qc, Rc, Ac, Bc, N=Nc, beta=self.beta)

        P1, F1, d1 = lq_aux.stationary_values()

        self.F_b = F1[:, 0:self.nh]
        self.F_f = F1[:, self.nh:]

        self.pihat = np.linalg.cholesky((self.pih.T @
            self.pih) + (self.beta @ self.thetah.T @ P1[0:self.nh, 0:self.nh] @ self.thetah)).T
        self.llambdahat = self.pihat @ self.F_b
        self.ubhat = - self.pihat @ self.F_f

        return
