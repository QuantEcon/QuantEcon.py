"""
Computes quantities associated with the Gaussian linear state space model.

"""

from textwrap import dedent
import numpy as np
from scipy.linalg import solve
from ._matrix_eqn import solve_discrete_lyapunov
from numba import jit
from .util import check_random_state


@jit(nopython=True)
def simulate_linear_model(A, x0, v, ts_length):
    r"""
    This is a separate function for simulating a vector linear system of
    the form

    .. math::

        x_{t+1} = A x_t + v_t

    given :math:`x_0` = x0

    Here :math:`x_t` and :math:`v_t` are both n x 1 and :math:`A` is n x n.

    The purpose of separating this functionality out is to target it for
    optimization by Numba.  For the same reason, matrix multiplication is
    broken down into for loops.

    Parameters
    ----------
    A : array_like or scalar(float)
        Should be n x n
    x0 : array_like
        Should be n x 1.  Initial condition
    v : np.ndarray
        Should be n x ts_length-1.  Its t-th column is used as the time t
        shock :math:`v_t`
    ts_length : int
        The length of the time series

    Returns
    -------
    x : np.ndarray
        Time series with ts_length columns, the t-th column being :math:`x_t`
    """
    A = np.asarray(A)
    n = A.shape[0]
    x = np.empty((n, ts_length))
    x[:, 0] = x0
    for t in range(ts_length-1):
        # x[:, t+1] = A.dot(x[:, t]) + v[:, t]
        for i in range(n):
            x[i, t+1] = v[i, t]                   # Shock
            for j in range(n):
                x[i, t+1] += A[i, j] * x[j, t]   # Dot Product
    return x


class LinearStateSpace:
    r"""
    A class that describes a Gaussian linear state space model of the
    form:

    .. math::

      x_{t+1} = A x_t + C w_{t+1}

      y_t = G x_t + H v_t

    where :math:`{w_t}` and :math:`{v_t}` are independent and standard normal
    with dimensions k and l respectively.  The initial conditions are
    :math:`\mu_0` and :math:`\Sigma_0` for :math:`x_0 \sim N(\mu_0, \Sigma_0)`.
    When :math:`\Sigma_0=0`, the draw of :math:`x_0` is exactly :math:`\mu_0`.

    Parameters
    ----------
    A : array_like or scalar(float)
        Part of the state transition equation.  It should be `n x n`
    C : array_like or scalar(float)
        Part of the state transition equation.  It should be `n x m`
    G : array_like or scalar(float)
        Part of the observation equation.  It should be `k x n`
    H : array_like or scalar(float), optional(default=None)
        Part of the observation equation.  It should be `k x l`
    mu_0 : array_like or scalar(float), optional(default=None)
        This is the mean of initial draw and is `n x 1`
    Sigma_0 : array_like or scalar(float), optional(default=None)
        This is the variance of the initial draw and is `n x n` and
        also should be positive definite and symmetric

    Attributes
    ----------
    A, C, G, H, mu_0, Sigma_0 : see Parameters
    n, k, m, l : scalar(int)
        The dimensions of x_t, y_t, w_t and v_t respectively

    """

    def __init__(self, A, C, G, H=None, mu_0=None, Sigma_0=None):
        self.A, self.G, self.C = list(map(self.convert, (A, G, C)))
        # = Check Input Shapes = #
        ni, nj = self.A.shape
        if ni != nj:
            raise ValueError(
                "Matrix A (shape: %s) needs to be square" % (self.A.shape, ))
        if ni != self.C.shape[0]:
            raise ValueError(
                "Matrix C (shape: %s) does not have compatible dimensions "
                "with A. It should be shape: %s" % (self.C.shape, (ni, 1)))
        self.m = self.C.shape[1]
        self.k, self.n = self.G.shape
        if self.n != ni:
            raise ValueError("Matrix G (shape: %s) does not have compatible"
                             "dimensions with A (%s)" % (self.G.shape,
                                                         self.A.shape))
        if H is None:
            self.H = None
            self.l = None
        else:
            self.H = self.convert(H)
            self.l = self.H.shape[1]
        if mu_0 is None:
            self.mu_0 = np.zeros((self.n, 1))
        else:
            self.mu_0 = self.convert(mu_0)
            self.mu_0.shape = self.n, 1
        if Sigma_0 is None:
            self.Sigma_0 = np.zeros((self.n, self.n))
        else:
            self.Sigma_0 = self.convert(Sigma_0)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        m = """\
        Linear Gaussian state space model:
          - dimension of state space          : {n}
          - number of innovations             : {m}
          - dimension of observation equation : {k}
        """
        return dedent(m.format(n=self.n, k=self.k, m=self.m))

    def convert(self, x):
        """
        Convert array_like objects (lists of lists, floats, etc.) into
        well formed 2D NumPy arrays

        """
        return np.atleast_2d(np.asarray(x, dtype='float'))

    def simulate(self, ts_length=100, random_state=None):
        r"""
        Simulate a time series of length ts_length, first drawing

        .. math::

            x_0 \sim N(\mu_0, \Sigma_0)

        Parameters
        ----------
        ts_length : scalar(int), optional(default=100)
            The length of the simulation
        random_state : int or np.random.RandomState/Generator, optional
            Random seed (integer) or np.random.RandomState or Generator
            instance to set the initial state of the random number
            generator for reproducibility. If None, a randomly
            initialized RandomState is used.

        Returns
        -------
        x : array_like(float)
            An n x ts_length array, where the t-th column is :math:`x_t`
        y : array_like(float)
            A k x ts_length array, where the t-th column is :math:`y_t`

        """
        random_state = check_random_state(random_state)

        x0 = random_state.multivariate_normal(self.mu_0.flatten(),
                                              self.Sigma_0)
        w = random_state.standard_normal((self.m, ts_length-1))
        v = self.C.dot(w)  # Multiply each w_t by C to get v_t = C w_t
        # == simulate time series == #
        x = simulate_linear_model(self.A, x0, v, ts_length)

        if self.H is not None:
            v = random_state.standard_normal((self.l, ts_length))
            y = self.G.dot(x) + self.H.dot(v)
        else:
            y = self.G.dot(x)

        return x, y

    def replicate(self, T=10, num_reps=100, random_state=None):
        r"""
        Simulate num_reps observations of :math:`x_T` and :math:`y_T` given
        :math:`x_0 \sim N(\mu_0, \Sigma_0)`.

        Parameters
        ----------
        T : scalar(int), optional(default=10)
            The period that we want to replicate values for
        num_reps : scalar(int), optional(default=100)
            The number of replications that we want
        random_state : int or np.random.RandomState/Generator, optional
            Random seed (integer) or np.random.RandomState or Generator
            instance to set the initial state of the random number
            generator for reproducibility. If None, a randomly
            initialized RandomState is used.

        Returns
        -------
        x : array_like(float)
            An n x num_reps array, where the j-th column is the j_th
            observation of :math:`x_T`

        y : array_like(float)
            A k x num_reps array, where the j-th column is the j_th
            observation of :math:`y_T`

        """
        random_state = check_random_state(random_state)

        x = np.empty((self.n, num_reps))
        for j in range(num_reps):
            x_T, _ = self.simulate(ts_length=T+1, random_state=random_state)
            x[:, j] = x_T[:, -1]
        if self.H is not None:
            v = random_state.standard_normal((self.l, num_reps))
            y = self.G.dot(x) + self.H.dot(v)
        else:
            y = self.G.dot(x)

        return x, y

    def moment_sequence(self):
        r"""
        Create a generator to calculate the population mean and
        variance-covariance matrix for both :math:`x_t` and :math:`y_t`
        starting at the initial condition (self.mu_0, self.Sigma_0).
        Each iteration produces a 4-tuple of items (mu_x, mu_y, Sigma_x,
        Sigma_y) for the next period.

        Yields
        ------
        mu_x : array_like(float)
            An n x 1 array representing the population mean of x_t
        mu_y : array_like(float)
            A  k x 1 array representing the population mean of y_t
        Sigma_x : array_like(float)
            An n x n array representing the variance-covariance matrix
            of x_t
        Sigma_y : array_like(float)
            A k x k array representing the variance-covariance matrix
            of y_t

        """
        # == Simplify names == #
        A, C, G, H = self.A, self.C, self.G, self.H
        # == Initial moments == #
        mu_x, Sigma_x = self.mu_0, self.Sigma_0

        while 1:
            mu_y = G.dot(mu_x)
            if H is None:
                Sigma_y = G.dot(Sigma_x).dot(G.T)
            else:
                Sigma_y = G.dot(Sigma_x).dot(G.T) + H.dot(H.T)

            yield mu_x, mu_y, Sigma_x, Sigma_y

            # == Update moments of x == #
            mu_x = A.dot(mu_x)
            Sigma_x = A.dot(Sigma_x).dot(A.T) + C.dot(C.T)

    def stationary_distributions(self):
        r"""
        Compute the moments of the stationary distributions of :math:`x_t` and
        :math:`y_t` if possible. Computation is by solving the discrete
        Lyapunov equation.

        Returns
        -------
        mu_x : array_like(float)
            An n x 1 array representing the stationary mean of :math:`x_t`
        mu_y : array_like(float)
            An k x 1 array representing the stationary mean of :math:`y_t`
        Sigma_x : array_like(float)
            An n x n array representing the stationary var-cov matrix
            of :math:`x_t`
        Sigma_y : array_like(float)
            An k x k array representing the stationary var-cov matrix
            of :math:`y_t`
        Sigma_yx : array_like(float)
            An k x n array representing the stationary cov matrix
            between :math:`y_t` and :math:`x_t`.

        """
        self.__partition()
        num_const, sorted_idx = self.num_const, self.sorted_idx
        A21, A22 = self.A21, self.A22
        CC2 = self.C2 @ self.C2.T
        n = self.n

        if num_const > 0:
            μ = solve(np.eye(n-num_const) - A22, A21)
        else:
            μ = solve(np.eye(n-num_const) - A22, np.zeros((n, 1)))
        Σ = solve_discrete_lyapunov(A22, CC2, method='bartels-stewart')

        mu_x = np.empty((n, 1))
        mu_x[:num_const] = self.mu_0[sorted_idx[:num_const]]
        mu_x[num_const:] = μ

        Sigma_x = np.zeros((n, n))
        Sigma_x[num_const:, num_const:] = Σ

        mu_x = self.P.T @ mu_x
        Sigma_x = self.P.T @ Sigma_x @ self.P

        mu_y = self.G @ mu_x
        Sigma_y = self.G @ Sigma_x @ self.G.T
        if self.H is not None:
            Sigma_y += self.H @ self.H.T
        Sigma_yx = self.G @ Sigma_x

        self.mu_x, self.mu_y = mu_x, mu_y
        self.Sigma_x, self.Sigma_y, self.Sigma_yx = Sigma_x, Sigma_y, Sigma_yx

        return mu_x, mu_y, Sigma_x, Sigma_y, Sigma_yx

    def geometric_sums(self, beta, x_t):
        r"""
        Forecast the geometric sums

        .. math::

            S_x := E \Big[ \sum_{j=0}^{\infty} \beta^j x_{t+j} | x_t \Big]

            S_y := E \Big[ \sum_{j=0}^{\infty} \beta^j y_{t+j} | x_t \Big]

        Parameters
        ----------
        beta : scalar(float)
            Discount factor, in [0, 1)

        beta : array_like(float)
            The term x_t for conditioning

        Returns
        -------
        S_x : array_like(float)
            Geometric sum as defined above

        S_y : array_like(float)
            Geometric sum as defined above

        """

        I = np.identity(self.n)
        S_x = solve(I - beta * self.A, x_t)
        S_y = self.G.dot(S_x)

        return S_x, S_y

    def impulse_response(self, j=5):
        r"""
        Pulls off the imuplse response coefficients to a shock
        in :math:`w_{t}` for :math:`x` and :math:`y`

        Important to note: We are uninterested in the shocks to
        v for this method

        * :math:`x` coefficients are :math:`C, AC, A^2 C...`
        * :math:`y` coefficients are :math:`GC, GAC, GA^2C...`

        Parameters
        ----------
        j : Scalar(int)
            Number of coefficients that we want

        Returns
        -------
        xcoef : list(array_like(float, 2))
            The coefficients for x
        ycoef : list(array_like(float, 2))
            The coefficients for y
        """
        # Pull out matrices
        A, C, G, H = self.A, self.C, self.G, self.H
        Apower = np.copy(A)

        # Create room for coefficients
        xcoef = [C]
        ycoef = [np.dot(G, C)]

        for i in range(j):
            xcoef.append(np.dot(Apower, C))
            ycoef.append(np.dot(G, np.dot(Apower, C)))
            Apower = np.dot(Apower, A)

        return xcoef, ycoef

    def __partition(self):
        r"""
        Reorder the states by shifting the constant terms to the
        top of the state vector. Then partition the linear state
        space model following Appendix C in RMT Ch2 such that the
        A22 matrix associated with non-constant states have eigenvalues
        all strictly smaller than 1.

        .. math::
            \left[\begin{array}{c}
            const\\
            x_{2,t+1}
            \end{array}\right]=\left[\begin{array}{cc}
            I & 0\\
            A_{21} & A_{22}
            \end{array}\right]\left[\begin{array}{c}
            1\\
            x_{2,t}
            \end{array}\right]+\left[\begin{array}{c}
            0\\
            C_{2}
            \end{array}\right]w_{t+1}

        Returns
        -------
        A22 : array_like(float)
            Lower right part of the reordered matrix A.
        A21 : array_like(float)
            Lower left part of the reordered matrix A.
        """
        A, C = self.A, self.C
        n = self.n

        sorted_idx = []
        A_diag = np.diag(A)
        num_const = 0
        for idx in range(n):
            if (A_diag[idx] == 1) and (C[idx, :] == 0).all() and \
                    np.linalg.norm(A[idx, :]) == 1:
                sorted_idx.insert(0, idx)
                num_const += 1
            else:
                sorted_idx.append(idx)
        self.num_const = num_const
        self.sorted_idx = sorted_idx

        P = np.zeros((n, n))
        P[range(n), sorted_idx] = 1

        sorted_A = P @ A @ P.T
        sorted_C = P @ C
        A21 = sorted_A[num_const:, :num_const]
        A22 = sorted_A[num_const:, num_const:]
        C2 = sorted_C[num_const:, :]

        self.P, self.A21, self.A22, self.C2 = P, A21, A22, C2

        return A21, A22
