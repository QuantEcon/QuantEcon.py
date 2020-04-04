"""
A module for working with additive and multiplicative functionals.

"""

import numpy as np
import scipy.linalg as la
import quantecon as qe
from collections import namedtuple
import warnings


ad_lss_var = namedtuple('additive_decomp', 'ν H g')
md_lss_var = namedtuple('multiplicative_decomp', 'ν_tilde H g')


class AMF_LSS_VAR:
    """
    A class for transforming an additive (multiplicative) functional into a
    QuantEcon linear state space system. It uses the first-order VAR
    representation to build the LSS representation using the
    `LinearStateSpace` class.

    First-order VAR representation:

    .. math::

        x_{t+1} = Ax_{t} + Bz_{t+1}

        y_{t+1}-y_{t} = ν + Dx_{t} + Fz_{t+1}

    Linear State Space (LSS) representation:

    .. math::

        \hat{x}_{t+1} = \hat{A}\hat{x}_{t}+\hat{B}z_{t+1}

        \hat{y}_{t} = \hat{D}\hat{x}_{t}

    Parameters
    ----------
    A : array_like(float, ndim=2)
        Part of the first-order vector autoregression equation. It should be an
        `nx x nx` matrix.

    B : array_like(float, ndim=2)
        Part of the first-order vector autoregression equation. It should be an
        `nx x nk` matrix.

    D : array_like(float, dim=2)
        Part of the nonstationary random process. It should be an `ny x nx`
        matrix.

    F : array_like or None, optional(default=None)
        Part of the nonstationary random process. If array_like, it should be
        an `ny x nk` matrix.

    ν : array_like or float or None, optional(default=None)
        Part of the nonstationary random process. If array_like, it should be
        an `ny x 1` matrix.

    Attributes
    ----------
    A, B, D, F, ν : See Parameters.

    additive_decomp : namedtuple
        A namedtuple containing the following items:
        ::

            "ν" : unconditional mean difference in Y
            "H" : coefficient for the (linear) martingale component
            "g" : coefficient for the stationary component g(x)

    multiplicative_decomp : namedtuple
        A namedtuple containing the following items:
        ::

            "ν_tilde" : eigenvalue
            "H" : coefficient for the (linear) martingale component
            "g" : coefficient for the stationary component g(x)

    """
    def __init__(self, A, B, D, F=None, ν=None):
        # = Set Inputs = #
        self.A = np.asarray(A)
        self.B = np.asarray(B)
        self._nx, self._nk = self.B.shape
        self.D = np.asarray(D)
        self._ny = self.D.shape[0]

        if hasattr(F, '__getitem__'):
            self.F = np.asarray(F)  # F is array_like
        else:
            self.F = np.zeros((self._nk, self._nk))

        if hasattr(ν, '__getitem__') or isinstance(ν, float):
            self.ν = np.asarray(ν)  # ν is array_like or float
        else:
            self.ν = np.zeros((self._ny, 1))

        # = Check dimensions = #
        self._attr_dims_check()

        # = Check shape = #
        self._attr_shape_check()

        # = Compute Additive Decomposition = #
        eye = np.identity(self._nx)
        A_res = la.solve(eye - self.A, eye)
        g = self.D @ A_res
        H = F + D @ A_res @ self.B

        self.additive_decomp = ad_lss_var(ν, H, g)

        # = Compute Multiplicative Decomposition = #
        ν_tilde = ν + (.5) * np.expand_dims(np.diag(H @ H.T), 1)
        self.multiplicative_decomp = md_lss_var(ν_tilde, H, g)

        # = Construct LSS = #
        nx0c = np.zeros((self._nx, 1))
        nx0r = np.zeros(self._nx)
        nx1 = np.ones(self._nx)
        nk0 = np.zeros(self._nk)
        ny0c = np.zeros((self._ny, 1))
        ny0r = np.zeros(self._ny)
        ny1m = np.eye(self._ny)
        ny0m = np.zeros((self._ny, self._ny))
        nyx0m = np.zeros_like(self.D)

        x0 = self._construct_x0(nx0r, ny0r)
        A_bar = self._construct_A_bar(x0, nx0c, nyx0m, ny0c, ny1m, ny0m)
        B_bar = self._construct_B_bar(nk0, H)
        G_Bar = self._construct_G_bar(nx0c, self._nx, nyx0m, ny0c, ny1m, ny0m,
                                      g)
        H_bar = self._construct_H_bar(self._nx, self._ny, self._nk)
        Sigma_0 = self._construct_Sigma_0(x0)

        self._lss = qe.LinearStateSpace(A_bar, B_bar, G_Bar, H_bar, mu_0=x0,
                                        Sigma_0=Sigma_0)

    def _construct_x0(self, nx0r, ny0r):
        x0 = np.hstack([1, 0, nx0r, ny0r, ny0r])

        return x0

    def _construct_A_bar(self, x0, nx0c, nyx0m, ny0c, ny1m, ny0m):
        # Build A matrix for LSS
        # Order of states is: [1, t, x_{t}, y_{t}, m_{t}]

        # Transition for 1
        A1 = x0.copy()

        # Transition for t
        A2 = x0.copy()
        A2[1] = 1.

        # Transition for x_{t+1}
        A3 = np.hstack([nx0c, nx0c, self.A, nyx0m.T, nyx0m.T])

        # Transition for y_{t+1}
        A4 = np.hstack([self.ν, ny0c, self.D, ny1m, ny0m])

        # Transition for m_{t+1}
        A5 = np.hstack([ny0c, ny0c, nyx0m, ny0m, ny1m])

        A_bar = np.vstack([A1, A2, A3, A4, A5])

        return A_bar

    def _construct_B_bar(self, nk0, H):
        # Build B matrix for LSS
        B_bar = np.vstack([nk0, nk0, self.B, self.F, H])

        return B_bar

    def _construct_G_bar(self, nx0c, nx, nyx0m, ny0c, ny1m, ny0m, g):
        # Build G matrix for LSS
        # Order of observation is: [x_{t}, y_{t}, m_{t}, s_{t}, tau_{t}]

        # Selector for x_{t}
        G1 = np.hstack([nx0c, nx0c, np.eye(nx), nyx0m.T, nyx0m.T])

        # Selector for y_{t}
        G2 = np.hstack([ny0c, ny0c, nyx0m, ny1m, ny0m])

        # Selector for martingale m_{t}
        G3 = np.hstack([ny0c, ny0c, nyx0m, ny0m, ny1m])

        # Selector for stationary s_{t}
        G4 = np.hstack([ny0c, ny0c, -g, ny0m, ny0m])

        # Selector for trend tau_{t}
        G5 = np.hstack([ny0c, self.ν, nyx0m, ny0m, ny0m])

        G_bar = np.vstack([G1, G2, G3, G4, G5])

        return G_bar

    def _construct_H_bar(self, nx, ny, nk):
        # Build H matrix for LSS
        H_bar = np.zeros((2 + nx + 2 * ny, nk))

        return H_bar

    def _construct_Sigma_0(self, x0):
        Sigma_0 = np.zeros((len(x0), len(x0)))

        return Sigma_0

    def _attr_dims_check(self):
        """Check the dimensions of attributes."""

        inputs = {'A': self.A, 'B': self.B, 'D': self.D, 'F': self.F,
                  'ν': self.ν}

        for input_name, input_val in inputs.items():
            if input_val.ndim != 2:
                raise ValueError(input_name + ' must have 2 dimensions.')

    def _attr_shape_check(self):
        """Check the shape of attributes."""

        same_dim_pairs = {'first': (0, {'A and B': [self.A, self.B],
                                        'D and F': [self.D, self.F],
                                        'D and ν': [self.D, self.ν]}),
                          'second': (1, {'A and D': [self.A, self.D],
                                         'B and F': [self.B, self.F]})}

        for dim_name, (dim_idx, pairs) in same_dim_pairs.items():
            for pair_name, (e0, e1) in pairs.items():
                if e0.shape[dim_idx] != e1.shape[dim_idx]:
                    raise ValueError('The ' + dim_name + ' dimensions of ' +
                                     pair_name + ' must match.')

        if self.A.shape[0] != self.A.shape[1]:
            raise ValueError('A (shape: %s) must be a square matrix.' %
                             (self.A.shape, ))

        # F.shape[0] == ν.shape[0] holds by transitivity
        # Same for D.shape[1] == B.shape[0] == A.shape[0]

    def loglikelihood_path(self, x, y):
        """
        Computes the log-likelihood path associated with a path of additive
        functionals `x` and `y` and assuming standard normal shocks.

        Parameters
        ----------
        x : ndarray(float, ndim=1)
            A path of observations for the state variable.

        y : ndarray(float, ndim=1)
            A path of observations for the random process

        Returns
        --------
        llh : ndarray(float, ndim=1)
            An array containing the loglikelihood path.

        """

        k, T = y.shape
        FF = self.F @ self.F.T
        FF_inv = la.inv(FF)
        temp = y[:, 1:] - y[:, :-1] - self.D @ x[:, :-1]
        obs = temp * FF_inv * temp
        obssum = np.cumsum(obs)
        scalar = (np.log(la.det(FF)) + k * np.log(2 * np.pi)) * np.arange(1, T)

        llh = (-0.5) * (obssum + scalar)

        return llh


def pth_order_to_stacked_1st_order(ζ_hat, A_hats):
    """
    Construct the first order stacked representation of a VAR from the pth
    order representation.

    Parameters
    ----------
    ζ_hat : ndarray(float, ndim=1)
        Vector of constants of the pth order VAR.

    A_hats : tuple
        Sequence of `ρ` matrices of shape `n x n` of lagged coefficients of
        the pth order VAR.

    Returns
    ----------
    ζ : ndarray(float, ndim=1)
        Vector of constants of the 1st order stacked VAR.

    A : ndarray(float, ndim=2)
        Matrix of coefficients of the 1st order stacked VAR.

    """
    ρ = len(A_hats)
    n = A_hats[0].shape[0]

    A = np.zeros((n * ρ, n * ρ))
    A[:n, :] = np.hstack(A_hats)
    A[n:, :n*(ρ-1)] = np.eye(n * (ρ - 1))

    ζ = np.zeros(n * ρ)
    ζ[:n] = np.eye(n) @ ζ_hat

    return ζ, A


def compute_BQ_restricted_B_0(A_hats, Ω_hat):
    """
    Compute the `B_0` matrix for `AMF_LSS_VAR` using the Blanchard and Quah
    method to impose long-run restrictions.

    Parameters
    ----------
    A_hats : tuple
        Sequence of `ρ` matrices of shape `n x n` of lagged coefficients of
        the pth order VAR.

    Ω_hat : ndarray(float, ndim=2)
        Covariance matrix of the error term.

    Returns
    ----------
    B_0 : ndarray(float, ndim=2)
        Matrix satisfying :math:`\hat{\Omega}=B_{0}B_{0}^{\intercal}`, where
        :math:`B_{0}` is identified using the Blanchard and Quah method.

    References
    ----------
    .. [1] Lars Peter Hansen and Thomas J. Sargent. Risk, Uncertainty, and
           Value. Princeton, New Jersey: Princeton University Press., 2018.

    """
    ρ = len(A_hats)

    # Step 1: Compute the spectral density of V_{t} at frequency zero
    def A_hat(z):
        return np.eye(ρ) - sum([A_hats[i] * z ** i for i in range(ρ)])
    A_hat_1 = A_hat(1)

    accuracy_loss = np.log10(np.linalg.cond(A_hat_1)).round().astype(int)
    if accuracy_loss >= 8:
        warnings.warn('The `A_hat(1)` matrix is ill-conditioned. ' +
                      ' Approximately ' + accuracy_loss + ' digits may be' +
                      ' lost due to matrix inversion.')

    A_hat_1_inv = np.linalg.inv(A_hat_1)
    R = A_hat_1_inv @ Ω_hat @ A_hat_1_inv.T

    # Step 2: Compute the Cholesky decomposition of R
    R_chol = np.linalg.cholesky(R)

    # Step 3: Compute B_0
    B_0 = A_hat_1 @ R_chol

    if not np.abs(B_0 @ B_0.T - Ω_hat).max() < 1e-10:
        raise ValueError('The process of identifying `B_0` failed.')

    return B_0
