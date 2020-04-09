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
    A, B, D, F, ν, nx, nk, ny : See Parameters.

    lss : Instance of `LinearStateSpace`.
        LSS representation of the additive (multiplicative) functional.

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

    Examples
    ----------
    Consider the following example:

    >>> ϕ_1, ϕ_2, ϕ_3, ϕ_4 = 0.5, -0.2, 0, 0.5
    >>> σ = 0.01
    >>> ν = 0.01   # Growth rate
    >>> A = np.array([[ϕ_1, ϕ_2, ϕ_3, ϕ_4],
    ...               [  1,   0,   0,   0],
    ...               [  0,   1,   0,   0],
    ...               [  0,   0,   1,   0]])
    >>> B = np.array([[σ, 0, 0, 0]]).T
    >>> D = np.array([[1, 0, 0, 0]]) @ A
    >>> F = np.array([[1, 0, 0, 0]]) @ B
    >>> amf = qe.AMF_LSS_VAR(A, B, D, F, ν=ν)

    The additive decomposition can be accessed by:

    >>> amf.multiplicative_decomp
    additive_decomp(ν=array([[0.01]]), H=array([[0.05]]),
    g=array([[4. , 1.5, 2.5, 2.5]]))

    The multiplicative decomposition can be accessed by:

    >>> amf.multiplicative_decomp
    multiplicative_decomp(ν_tilde=array([[0.01125]]), H=array([[0.05]]),
    g=array([[4. , 1.5, 2.5, 2.5]]))

    References
    ----------
    .. [1] Lars Peter Hansen and Thomas J Sargent. Robustness. Princeton
       university press, 2008.

    .. [2] Lars Peter Hansen and José A Scheinkman. Long-term risk: An operator
       approach. Econometrica, 77(1):177–234, 2009.

    """
    def __init__(self, A, B, D, F=None, ν=None):
        # = Set Inputs = #
        self.A = np.atleast_2d(A)
        self.B = np.atleast_2d(B)
        self.nx, self.nk = self.B.shape
        self.D = np.atleast_2d(D)
        self.ny = self.D.shape[0]

        if hasattr(F, '__getitem__'):
            self.F = np.atleast_2d(F)  # F is array_like
        else:
            self.F = np.zeros((self.nk, self.nk))

        if hasattr(ν, '__getitem__') or isinstance(ν, float):
            self.ν = np.atleast_2d(ν)  # ν is array_like or float
        else:
            self.ν = np.zeros((self.ny, 1))

        # = Check dimensions = #
        self._attr_dims_check()

        # = Check shape = #
        self._attr_shape_check()

        # = Compute Additive Decomposition = #
        eye = np.identity(self.nx)
        A_res = la.solve(eye - self.A, eye)
        g = self.D @ A_res
        H = self.F + self.D @ A_res @ self.B

        self.additive_decomp = ad_lss_var(self.ν, H, g)

        # = Compute Multiplicative Decomposition = #
        ν_tilde = self.ν + (.5) * np.expand_dims(np.diag(H @ H.T), 1)
        self.multiplicative_decomp = md_lss_var(ν_tilde, H, g)

        # = Construct LSS = #
        nx0c = np.zeros((self.nx, 1))
        nx0r = np.zeros(self.nx)
        nx1 = np.ones(self.nx)
        nk0 = np.zeros(self.nk)
        ny0c = np.zeros((self.ny, 1))
        ny0r = np.zeros(self.ny)
        ny1m = np.eye(self.ny)
        ny0m = np.zeros((self.ny, self.ny))
        nyx0m = np.zeros_like(self.D)

        x0 = self._construct_x0(nx0r, ny0r)
        A_bar = self._construct_A_bar(x0, nx0c, nyx0m, ny0c, ny1m, ny0m)
        B_bar = self._construct_B_bar(nk0, H)
        G_bar = self._construct_G_bar(nx0c, self.nx, nyx0m, ny0c, ny1m, ny0m,
                                      g)
        H_bar = self._construct_H_bar(self.nx, self.ny, self.nk)
        Sigma_0 = self._construct_Sigma_0(x0)

        self.lss = qe.LinearStateSpace(A_bar, B_bar, G_bar, H_bar, mu_0=x0,
                                       Sigma_0=Sigma_0)

    def _construct_x0(self, nx0r, ny0r):
        "Construct initial state x0 for LSS instance."

        x0 = np.hstack([1, 0, nx0r, ny0r, ny0r])

        return x0

    def _construct_A_bar(self, x0, nx0c, nyx0m, ny0c, ny1m, ny0m):
        "Construct A matrix for LSS instance."

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
        "Construct B matrix for LSS instance."
        B_bar = np.vstack([nk0, nk0, self.B, self.F, H])

        return B_bar

    def _construct_G_bar(self, nx0c, nx, nyx0m, ny0c, ny1m, ny0m, g):
        "Construct G matrix for LSS instance."

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
        "Construct H matrix for LSS instance."

        H_bar = np.zeros((2 + nx + 2 * ny, nk))

        return H_bar

    def _construct_Sigma_0(self, x0):
        "Construct initial covariance matrix Sigma_0 for LSS instance."

        Sigma_0 = np.zeros((len(x0), len(x0)))

        return Sigma_0

    def _attr_dims_check(self):
        "Check the dimensions of attributes."

        inputs = {'A': self.A, 'B': self.B, 'D': self.D, 'F': self.F,
                  'ν': self.ν}

        for input_name, input_val in inputs.items():
            if input_val.ndim != 2:
                raise ValueError(input_name + ' must have 2 dimensions.')

    def _attr_shape_check(self):
        "Check the shape of attributes."

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
