"""
Defining various quadrature routines.

Based on the quadrature routines found in the CompEcon toolbox by
Miranda and Fackler.

References
----------
Miranda, Mario J, and Paul L Fackler. Applied Computational Economics
and Finance, MIT Press, 2002.

"""
import math
import numpy as np
import scipy.linalg as la
from numba import jit, vectorize
from .ce_util import ckron, gridmake
from .util import check_random_state

__all__ = ['qnwcheb', 'qnwequi', 'qnwlege', 'qnwnorm', 'qnwlogn',
           'qnwsimp', 'qnwtrap', 'qnwunif', 'quadrect', 'qnwbeta',
           'qnwgamma']


@vectorize(nopython=True)
def gammaln(x):
    return math.lgamma(x)


@vectorize(nopython=True)
def fix(x):
    if x < 0:
        return math.ceil(x)
    else:
        return math.floor(x)


# ------------------ #
# Exported Functions #
# ------------------ #


def qnwcheb(n, a=1, b=1):
    """
    Computes multivariate Guass-Checbychev quadrature nodes and weights.

    Parameters
    ----------
    n : int or array_like(float)
        A length-d iterable of the number of nodes in each dimension

    a : scalar or array_like(float)
        A length-d iterable of lower endpoints. If a scalar is given,
        that constant is repeated d times, where d is the number of
        dimensions

    b : scalar or array_like(float)
        A length-d iterable of upper endpoints. If a scalar is given,
        that constant is repeated d times, where d is the number of
        dimensions

    Returns
    -------
    nodes : np.ndarray(dtype=float)
        Quadrature nodes

    weights : np.ndarray(dtype=float)
        Weights for quadrature nodes

    Notes
    -----
    Based of original function ``qnwcheb`` in CompEcon toolbox by
    Miranda and Fackler

    References
    ----------
    Miranda, Mario J, and Paul L Fackler. Applied Computational
    Economics and Finance, MIT Press, 2002.

    """
    return _make_multidim_func(_qnwcheb1, n, a, b)


def qnwequi(n, a, b, kind="N", equidist_pp=None, random_state=None):
    """
    Generates equidistributed sequences with property that averages
    value of integrable function evaluated over the sequence converges
    to the integral as n goes to infinity.

    Parameters
    ----------
    n : int
        Number of sequence points

    a : scalar or array_like(float)
        A length-d iterable of lower endpoints. If a scalar is given,
        that constant is repeated d times, where d is the number of
        dimensions

    b : scalar or array_like(float)
        A length-d iterable of upper endpoints. If a scalar is given,
        that constant is repeated d times, where d is the number of
        dimensions

    kind : string, optional(default="N")
        One of the following:

        - N - Neiderreiter (default)
        - W - Weyl
        - H - Haber
        - R - pseudo Random

    equidist_pp : array_like, optional(default=None)
        TODO: I don't know what this does

    random_state : int or np.random.RandomState, optional
        Random seed (integer) or np.random.RandomState instance to set
        the initial state of the random number generator for
        reproducibility. If None, a randomly initialized RandomState is
        used.

    Returns
    -------
    nodes : np.ndarray(dtype=float)
        Quadrature nodes

    weights : np.ndarray(dtype=float)
        Weights for quadrature nodes

    Notes
    -----
    Based of original function ``qnwequi`` in CompEcon toolbox by
    Miranda and Fackler

    References
    ----------
    Miranda, Mario J, and Paul L Fackler. Applied Computational
    Economics and Finance, MIT Press, 2002.

    """
    random_state = check_random_state(random_state)

    if equidist_pp is None:
        import sympy as sym
        equidist_pp = np.sqrt(np.array(list(sym.primerange(0, 7920))))

    n, a, b = list(map(np.atleast_1d, list(map(np.asarray, [n, a, b]))))

    d = max(list(map(len, [n, a, b])))
    n = np.prod(n)

    if a.size == 1:
        a = np.repeat(a, d)

    if b.size == 1:
        b = np.repeat(b, d)

    # Specify `dtype=np.int64` to avoid overflow on Windows
    i = np.arange(1, n + 1, dtype=np.int64)

    if kind.upper() == "N":  # Neiderreiter
        j = 2.0 ** (np.arange(1, d+1) / (d+1))
        nodes = np.outer(i, j)
        nodes = (nodes - fix(nodes)).squeeze()
    elif kind.upper() == "W":  # Weyl
        j = equidist_pp[:d]
        nodes = np.outer(i, j)
        nodes = (nodes - fix(nodes)).squeeze()
    elif kind.upper() == "H":  # Haber
        j = equidist_pp[:d]
        nodes = np.outer(i * (i+1) / 2, j)
        nodes = (nodes - fix(nodes)).squeeze()
    elif kind.upper() == "R":  # pseudo-random
        nodes = random_state.rand(n, d).squeeze()
    else:
        raise ValueError("Unknown sequence requested")

    # compute nodes and weights
    r = b - a
    nodes = a + nodes * r
    weights = (np.prod(r) / n) * np.ones(n)

    return nodes, weights


def qnwlege(n, a, b):
    """
    Computes multivariate Guass-Legendre  quadrature nodes and weights.

    Parameters
    ----------
    n : int or array_like(float)
        A length-d iterable of the number of nodes in each dimension

    a : scalar or array_like(float)
        A length-d iterable of lower endpoints. If a scalar is given,
        that constant is repeated d times, where d is the number of
        dimensions

    b : scalar or array_like(float)
        A length-d iterable of upper endpoints. If a scalar is given,
        that constant is repeated d times, where d is the number of
        dimensions

    Returns
    -------
    nodes : np.ndarray(dtype=float)
        Quadrature nodes

    weights : np.ndarray(dtype=float)
        Weights for quadrature nodes

    Notes
    -----
    Based of original function ``qnwlege`` in CompEcon toolbox by
    Miranda and Fackler

    References
    ----------
    Miranda, Mario J, and Paul L Fackler. Applied Computational
    Economics and Finance, MIT Press, 2002.

    """
    return _make_multidim_func(_qnwlege1, n, a, b)


def qnwnorm(n, mu=None, sig2=None, usesqrtm=False):
    """
    Computes nodes and weights for multivariate normal distribution

    Parameters
    ----------
    n : int or array_like(float)
        A length-d iterable of the number of nodes in each dimension

    mu : scalar or array_like(float), optional(default=zeros(d))
        The means of each dimension of the random variable. If a scalar
        is given, that constant is repeated d times, where d is the
        number of dimensions

    sig2 : array_like(float), optional(default=eye(d))
        A d x d array representing the variance-covariance matrix of the
        multivariate normal distribution.

    Returns
    -------
    nodes : np.ndarray(dtype=float)
        Quadrature nodes

    weights : np.ndarray(dtype=float)
        Weights for quadrature nodes

    Notes
    -----
    Based of original function ``qnwnorm`` in CompEcon toolbox by
    Miranda and Fackler

    References
    ----------
    Miranda, Mario J, and Paul L Fackler. Applied Computational
    Economics and Finance, MIT Press, 2002.

    """
    n = np.atleast_1d(n)
    d = n.size

    if mu is None:
        mu = np.zeros(d)
    else:
        mu = np.atleast_1d(mu)

    if sig2 is None:
        sig2 = np.eye(d)
    else:
        sig2 = np.atleast_1d(sig2).reshape(d, d)

    if all([x.size == 1 for x in [n, mu, sig2]]):
        nodes, weights = _qnwnorm1(n[0])
    else:
        nodes = []
        weights = []

        for i in range(d):
            _1d = _qnwnorm1(n[i])
            nodes.append(_1d[0])
            weights.append(_1d[1])

        nodes = gridmake(*nodes)
        weights = ckron(*weights[::-1])

    if usesqrtm:
        new_sig2 = la.sqrtm(sig2)
    else:  # cholesky
        new_sig2 = la.cholesky(sig2)

    if d > 1:
        nodes = nodes.dot(new_sig2) + mu  # Broadcast ok
    else:  # nodes.dot(sig) will not be aligned in scalar case.
        nodes = nodes * new_sig2 + mu

    return nodes.squeeze(), weights


def qnwlogn(n, mu=None, sig2=None):
    """
    Computes nodes and weights for multivariate lognormal distribution

    Parameters
    ----------
    n : int or array_like(float)
        A length-d iterable of the number of nodes in each dimension

    mu : scalar or array_like(float), optional(default=zeros(d))
        The means of each dimension of the random variable. If a scalar
        is given, that constant is repeated d times, where d is the
        number of dimensions

    sig2 : array_like(float), optional(default=eye(d))
        A d x d array representing the variance-covariance matrix of the
        multivariate normal distribution.

    Returns
    -------
    nodes : np.ndarray(dtype=float)
        Quadrature nodes

    weights : np.ndarray(dtype=float)
        Weights for quadrature nodes

    Notes
    -----
    Based of original function ``qnwlogn`` in CompEcon toolbox by
    Miranda and Fackler

    References
    ----------
    Miranda, Mario J, and Paul L Fackler. Applied Computational
    Economics and Finance, MIT Press, 2002.

    """
    nodes, weights = qnwnorm(n, mu, sig2)
    return np.exp(nodes), weights


def qnwsimp(n, a, b):
    """
    Computes multivariate Simpson quadrature nodes and weights.

    Parameters
    ----------
    n : int or array_like(float)
        A length-d iterable of the number of nodes in each dimension

    a : scalar or array_like(float)
        A length-d iterable of lower endpoints. If a scalar is given,
        that constant is repeated d times, where d is the number of
        dimensions

    b : scalar or array_like(float)
        A length-d iterable of upper endpoints. If a scalar is given,
        that constant is repeated d times, where d is the number of
        dimensions

    Returns
    -------
    nodes : np.ndarray(dtype=float)
        Quadrature nodes

    weights : np.ndarray(dtype=float)
        Weights for quadrature nodes

    Notes
    -----
    Based of original function ``qnwsimp`` in CompEcon toolbox by
    Miranda and Fackler

    References
    ----------
    Miranda, Mario J, and Paul L Fackler. Applied Computational
    Economics and Finance, MIT Press, 2002.

    """
    return _make_multidim_func(_qnwsimp1, n, a, b)


def qnwtrap(n, a, b):
    """
    Computes multivariate trapezoid rule quadrature nodes and weights.

    Parameters
    ----------
    n : int or array_like(float)
        A length-d iterable of the number of nodes in each dimension

    a : scalar or array_like(float)
        A length-d iterable of lower endpoints. If a scalar is given,
        that constant is repeated d times, where d is the number of
        dimensions

    b : scalar or array_like(float)
        A length-d iterable of upper endpoints. If a scalar is given,
        that constant is repeated d times, where d is the number of
        dimensions

    Returns
    -------
    nodes : np.ndarray(dtype=float)
        Quadrature nodes

    weights : np.ndarray(dtype=float)
        Weights for quadrature nodes

    Notes
    -----
    Based of original function ``qnwtrap`` in CompEcon toolbox by
    Miranda and Fackler

    References
    ----------
    Miranda, Mario J, and Paul L Fackler. Applied Computational
    Economics and Finance, MIT Press, 2002.

    """
    return _make_multidim_func(_qnwtrap1, n, a, b)


def qnwunif(n, a, b):
    """
    Computes quadrature nodes and weights for multivariate uniform
    distribution

    Parameters
    ----------
    n : int or array_like(float)
        A length-d iterable of the number of nodes in each dimension

    a : scalar or array_like(float)
        A length-d iterable of lower endpoints. If a scalar is given,
        that constant is repeated d times, where d is the number of
        dimensions

    b : scalar or array_like(float)
        A length-d iterable of upper endpoints. If a scalar is given,
        that constant is repeated d times, where d is the number of
        dimensions

    Returns
    -------
    nodes : np.ndarray(dtype=float)
        Quadrature nodes

    weights : np.ndarray(dtype=float)
        Weights for quadrature nodes

    Notes
    -----
    Based of original function ``qnwunif`` in CompEcon toolbox by
    Miranda and Fackler

    References
    ----------
    Miranda, Mario J, and Paul L Fackler. Applied Computational
    Economics and Finance, MIT Press, 2002.

    """
    n, a, b = list(map(np.asarray, [n, a, b]))
    nodes, weights = qnwlege(n, a, b)
    weights = weights / np.prod(b - a)
    return nodes, weights


def quadrect(f, n, a, b, kind='lege', *args, **kwargs):
    """
    Integrate the d-dimensional function f on a rectangle with lower and
    upper bound for dimension i defined by a[i] and b[i], respectively;
    using n[i] points.

    Parameters
    ----------
    f : function
        The function to integrate over. This should be a function
        that accepts as its first argument a matrix representing points
        along each dimension (each dimension is a column). Other
        arguments that need to be passed to the function are caught by
        `*args` and `**kwargs`

    n : int or array_like(float)
        A length-d iterable of the number of nodes in each dimension

    a : scalar or array_like(float)
        A length-d iterable of lower endpoints. If a scalar is given,
        that constant is repeated d times, where d is the number of
        dimensions

    b : scalar or array_like(float)
        A length-d iterable of upper endpoints. If a scalar is given,
        that constant is repeated d times, where d is the number of
        dimensions

    kind : string, optional(default='lege')
        Specifies which type of integration to perform. Valid
        values are:

        lege - Gauss-Legendre
        cheb - Gauss-Chebyshev
        trap - trapezoid rule
        simp - Simpson rule
        N    - Neiderreiter equidistributed sequence
        W    - Weyl equidistributed sequence
        H    - Haber  equidistributed sequence
        R    - Monte Carlo

    *args, **kwargs :
        Other arguments passed to the function f

    Returns
    -------
    out : scalar (float)
        The value of the integral on the region [a, b]

    Notes
    -----
    Based of original function ``quadrect`` in CompEcon toolbox by
    Miranda and Fackler

    References
    ----------
    Miranda, Mario J, and Paul L Fackler. Applied Computational
    Economics and Finance, MIT Press, 2002.

    """
    if kind.lower() == "lege":
        nodes, weights = qnwlege(n, a, b)
    elif kind.lower() == "cheb":
        nodes, weights = qnwcheb(n, a, b)
    elif kind.lower() == "trap":
        nodes, weights = qnwtrap(n, a, b)
    elif kind.lower() == "simp":
        nodes, weights = qnwsimp(n, a, b)
    else:
        nodes, weights = qnwequi(n, a, b, kind)

    out = weights.dot(f(nodes, *args, **kwargs))
    return out


def qnwbeta(n, a=1.0, b=1.0):
    """
    Computes nodes and weights for beta distribution

    Parameters
    ----------
    n : int or array_like(float)
        A length-d iterable of the number of nodes in each dimension

    a : scalar or array_like(float), optional(default=1.0)
        A length-d

    b : array_like(float), optional(default=1.0)
        A d x d array representing the variance-covariance matrix of the
        multivariate normal distribution.

    Returns
    -------
    nodes : np.ndarray(dtype=float)
        Quadrature nodes

    weights : np.ndarray(dtype=float)
        Weights for quadrature nodes

    Notes
    -----
    Based of original function ``qnwbeta`` in CompEcon toolbox by
    Miranda and Fackler

    References
    ----------
    Miranda, Mario J, and Paul L Fackler. Applied Computational
    Economics and Finance, MIT Press, 2002.

    """
    return _make_multidim_func(_qnwbeta1, n, a, b)


def qnwgamma(n, a=1.0, b=1.0, tol=3e-14):
    """
    Computes nodes and weights for gamma distribution

    Parameters
    ----------
    n : int or array_like(float)
        A length-d iterable of the number of nodes in each dimension

    a : scalar or array_like(float) : optional(default=ones(d))
        Shape parameter of the gamma distribution parameter. Must be positive

    b : scalar or array_like(float) : optional(default=ones(d))
        Scale parameter of the gamma distribution parameter. Must be positive

    tol : scalar or array_like(float) : optional(default=ones(d) * 3e-14)
        Tolerance parameter for newton iterations for each node

    Returns
    -------
    nodes : np.ndarray(dtype=float)
        Quadrature nodes

    weights : np.ndarray(dtype=float)
        Weights for quadrature nodes

    Notes
    -----
    Based of original function ``qnwgamma`` in CompEcon toolbox by
    Miranda and Fackler

    References
    ----------
    Miranda, Mario J, and Paul L Fackler. Applied Computational
    Economics and Finance, MIT Press, 2002.

    """
    return _make_multidim_func(_qnwgamma1, n, a, b, tol)

# ------------------ #
# Internal Functions #
# ------------------ #


def _make_multidim_func(one_d_func, n, *args):
    """
    A helper function to cut down on code repetition. Almost all of the
    code in qnwcheb, qnwlege, qnwsimp, qnwtrap is just dealing
    various forms of input arguments and then shelling out to the
    corresponding 1d version of the function.

    This routine does all the argument checking and passes things
    through the appropriate 1d function before using a tensor product
    to combine weights and nodes.

    Parameters
    ----------
    one_d_func : function
        The 1d function to be called along each dimension

    n : int or array_like(float)
        A length-d iterable of the number of nodes in each dimension

    args :
        These are the arguments to various qnw____ functions.  For the
        majority of the functions this is just a and b, but some differ.

    Returns
    -------
    func : function
        The multi-dimensional version of the parameter ``one_d_func``


    """
    _args = list(args)
    n = np.atleast_1d(n)
    args = list(map(np.atleast_1d, _args))

    if all([x.size == 1 for x in [n] + args]):
        return one_d_func(n[0], *_args)

    d = n.size

    for i in range(len(args)):
        if args[i].size == 1:
            args[i] = np.repeat(args[i], d)

    nodes = []
    weights = []

    for i in range(d):
        ai = [x[i] for x in args]
        _1d = one_d_func(n[i], *ai)
        nodes.append(_1d[0])
        weights.append(_1d[1])

    weights = ckron(*weights[::-1])  # reverse ordered tensor product

    nodes = gridmake(*nodes)
    return nodes, weights


@jit(nopython=True)
def _qnwcheb1(n, a, b):
    """
    Compute univariate Guass-Checbychev quadrature nodes and weights

    Parameters
    ----------
    n : int
        The number of nodes

    a : int
        The lower endpoint

    b : int
        The upper endpoint

    Returns
    -------
    nodes : np.ndarray(dtype=float)
        An n element array of nodes

    nodes : np.ndarray(dtype=float)
        An n element array of weights

    Notes
    -----
    Based of original function ``qnwcheb1`` in CompEcon toolbox by
    Miranda and Fackler

    References
    ----------
    Miranda, Mario J, and Paul L Fackler. Applied Computational
    Economics and Finance, MIT Press, 2002.

    """
    nodes = (b+a)/2 - (b-a)/2 * np.cos(np.pi/n * np.linspace(0.5, n-0.5, n))

    # Create temporary arrays to be used in computing weights
    t1 = np.arange(1, n+1) - 0.5
    t2 = np.arange(0.0, n, 2)
    t3 = np.concatenate((np.array([1.0]),
                        -2.0/(np.arange(1.0, n-1, 2)*np.arange(3.0, n+1, 2))))

    # compute weights and return
    weights = ((b-a)/n)*np.cos(np.pi/n*np.outer(t1, t2)) @ t3

    return nodes, weights


@jit(nopython=True)
def _qnwlege1(n, a, b):
    """
    Compute univariate Guass-Legendre quadrature nodes and weights

    Parameters
    ----------
    n : int
        The number of nodes

    a : int
        The lower endpoint

    b : int
        The upper endpoint

    Returns
    -------
    nodes : np.ndarray(dtype=float)
        An n element array of nodes

    nodes : np.ndarray(dtype=float)
        An n element array of weights

    Notes
    -----
    Based of original function ``qnwlege1`` in CompEcon toolbox by
    Miranda and Fackler

    References
    ----------
    Miranda, Mario J, and Paul L Fackler. Applied Computational
    Economics and Finance, MIT Press, 2002.

    """
    # import ipdb; ipdb.set_trace()
    maxit = 100
    m = int(fix((n + 1) / 2.0))
    xm = 0.5 * (b + a)
    xl = 0.5 * (b - a)
    nodes = np.zeros(n)

    weights = nodes.copy()
    i = np.arange(m)

    z = np.cos(np.pi * ((i + 1.0) - 0.25) / (n + 0.5))

    for its in range(maxit):
        p1 = np.ones_like(z)
        p2 = np.zeros_like(z)
        for j in range(1, n+1):
            p3 = p2
            p2 = p1
            p1 = ((2 * j - 1) * z * p2 - (j - 1) * p3) / j

        # https://github.com/QuantEcon/QuantEcon.py/issues/530
        top = n * (z * p1 - p2)
        bottom = z ** 2 - 1.0
        pp = top / bottom
        z1 = z.copy()
        z = z1 - p1/pp
        if np.all(np.abs(z - z1) < 1e-14):
            break

    if its == maxit - 1:
        raise ValueError("Maximum iterations in _qnwlege1")

    nodes[i] = xm - xl * z
    nodes[- i - 1] = xm + xl * z

    # https://github.com/QuantEcon/QuantEcon.py/issues/530
    weights[i] = 2 * xl / ((1 - z ** 2) * pp * pp)
    weights[- i - 1] = weights[i]

    return nodes, weights


@jit(nopython=True)
def _qnwnorm1(n):
    """
    Compute nodes and weights for quadrature of univariate standard
    normal distribution

    Parameters
    ----------
    n : int
        The number of nodes

    Returns
    -------
    nodes : np.ndarray(dtype=float)
        An n element array of nodes

    nodes : np.ndarray(dtype=float)
        An n element array of weights

    Notes
    -----
    Based of original function ``qnwnorm1`` in CompEcon toolbox by
    Miranda and Fackler

    References
    ----------
    Miranda, Mario J, and Paul L Fackler. Applied Computational
    Economics and Finance, MIT Press, 2002.

    """
    maxit = 100
    pim4 = 1 / np.pi**(0.25)
    m = int(fix((n + 1) / 2))
    nodes = np.zeros(n)
    weights = np.zeros(n)

    for i in range(m):
        if i == 0:
            z = np.sqrt(2*n+1) - 1.85575 * ((2 * n + 1)**(-1 / 6.1))
        elif i == 1:
            z = z - 1.14 * (n ** 0.426) / z
        elif i == 2:
            z = 1.86 * z + 0.86 * nodes[0]
        elif i == 3:
            z = 1.91 * z + 0.91 * nodes[1]
        else:
            z = 2 * z + nodes[i-2]

        its = 0

        while its < maxit:
            its += 1
            p1 = pim4
            p2 = 0
            for j in range(1, n+1):
                p3 = p2
                p2 = p1
                p1 = z * math.sqrt(2.0/j) * p2 - math.sqrt((j - 1.0) / j) * p3

            pp = math.sqrt(2 * n) * p2
            z1 = z
            z = z1 - p1/pp
            if abs(z - z1) < 1e-14:
                break

        if its == maxit:
            raise ValueError("Failed to converge in _qnwnorm1")

        nodes[n - 1 - i] = z
        nodes[i] = -z
        weights[i] = 2 / (pp*pp)
        weights[n - 1 - i] = weights[i]

    weights /= math.sqrt(math.pi)
    nodes = nodes * math.sqrt(2.0)

    return nodes, weights


@jit(nopython=True)
def _qnwsimp1(n, a, b):
    """
    Compute univariate Simpson quadrature nodes and weights

    Parameters
    ----------
    n : int
        The number of nodes

    a : int
        The lower endpoint

    b : int
        The upper endpoint

    Returns
    -------
    nodes : np.ndarray(dtype=float)
        An n element array of nodes

    nodes : np.ndarray(dtype=float)
        An n element array of weights

    Notes
    -----
    Based of original function ``qnwsimp1`` in CompEcon toolbox by
    Miranda and Fackler

    References
    ----------
    Miranda, Mario J, and Paul L Fackler. Applied Computational
    Economics and Finance, MIT Press, 2002.

    """
    if n % 2 == 0:
        print("WARNING qnwsimp: n must be an odd integer. Increasing by 1")
        n += 1

    nodes = np.linspace(a, b, n)
    dx = nodes[1] - nodes[0]
    weights = np.kron(np.ones((n+1) // 2), np.array([2.0, 4.0]))
    weights = weights[:n]
    weights[0] = weights[-1] = 1
    weights = (dx / 3.0) * weights

    return nodes, weights


@jit(nopython=True)
def _qnwtrap1(n, a, b):
    """
    Compute univariate trapezoid rule quadrature nodes and weights

    Parameters
    ----------
    n : int
        The number of nodes

    a : int
        The lower endpoint

    b : int
        The upper endpoint

    Returns
    -------
    nodes : np.ndarray(dtype=float)
        An n element array of nodes

    nodes : np.ndarray(dtype=float)
        An n element array of weights

    Notes
    -----
    Based of original function ``qnwtrap1`` in CompEcon toolbox by
    Miranda and Fackler

    References
    ----------
    Miranda, Mario J, and Paul L Fackler. Applied Computational
    Economics and Finance, MIT Press, 2002.

    """
    if n < 1:
        raise ValueError("n must be at least one")

    nodes = np.linspace(a, b, n)
    dx = nodes[1] - nodes[0]

    weights = dx * np.ones(n)
    weights[0] *= 0.5
    weights[-1] *= 0.5

    return nodes, weights


@jit(nopython=True)
def _qnwbeta1(n, a=1.0, b=1.0):
    """
    Computes nodes and weights for quadrature on the beta distribution.
    Default is a=b=1 which is just a uniform distribution

    NOTE: For now I am just following compecon; would be much better to
    find a different way since I don't know what they are doing.

    Parameters
    ----------
    n : scalar : int
        The number of quadrature points

    a : scalar : float, optional(default=1)
        First Beta distribution parameter

    b : scalar : float, optional(default=1)
        Second Beta distribution parameter

    Returns
    -------
    nodes : np.ndarray(dtype=float, ndim=1)
        The quadrature points

    weights : np.ndarray(dtype=float, ndim=1)
        The quadrature weights that correspond to nodes

    Notes
    -----
    Based of original function ``_qnwbeta1`` in CompEcon toolbox by
    Miranda and Fackler

    References
    ----------
    Miranda, Mario J, and Paul L Fackler. Applied Computational
    Economics and Finance, MIT Press, 2002.

    """
    # We subtract one and write a + 1 where we actually want a, and a
    # where we want a - 1
    a = a - 1
    b = b - 1

    maxiter = 25

    # Allocate empty space
    nodes = np.zeros(n)
    weights = np.zeros(n)

    # Find "reasonable" starting values.  Why these numbers?
    for i in range(n):
        if i == 0:
            an = a/n
            bn = b/n
            r1 = (1+a) * (2.78/(4+n*n) + .768*an/n)
            r2 = 1 + 1.48*an + .96*bn + .452*an*an + .83*an*bn
            z = 1 - r1/r2
        elif i == 1:
            r1 = (4.1+a) / ((1+a)*(1+0.156*a))
            r2 = 1 + 0.06 * (n-8) * (1+0.12*a)/n
            r3 = 1 + 0.012*b * (1+0.25*abs(a))/n
            z = z - (1-z) * r1 * r2 * r3
        elif i == 2:
            r1 = (1.67+0.28*a)/(1+0.37*a)
            r2 = 1+0.22*(n-8)/n
            r3 = 1+8*b/((6.28+b)*n*n)
            z = z-(nodes[0]-z)*r1*r2*r3
        elif i == n - 2:
            r1 = (1+0.235*b)/(0.766+0.119*b)
            r2 = 1/(1+0.639*(n-4)/(1+0.71*(n-4)))
            r3 = 1/(1+20*a/((7.5+a)*n*n))
            z = z+(z-nodes[-4])*r1*r2*r3
        elif i == n - 1:
            r1 = (1+0.37*b) / (1.67+0.28*b)
            r2 = 1 / (1+0.22*(n-8)/n)
            r3 = 1 / (1+8*a/((6.28+a)*n*n))
            z = z+(z-nodes[-3])*r1*r2*r3
        else:
            z = 3*nodes[i-1] - 3*nodes[i-2] + nodes[i-3]

        ab = a+b

        # Root finding
        its = 0
        z1 = -100
        while abs(z - z1) > 1e-10 and its < maxiter:
            temp = 2 + ab
            p1 = (a-b + temp*z)/2
            p2 = 1

            for j in range(2, n+1):
                p3 = p2
                p2 = p1
                temp = 2*j + ab
                aa = 2*j * (j+ab)*(temp-2)
                bb = (temp-1) * (a*a - b*b + temp*(temp-2) * z)
                c = 2 * (j - 1 + a) * (j - 1 + b) * temp
                p1 = (bb*p2 - c*p3)/aa

            pp = (n*(a-b-temp*z) * p1 + 2*(n+a)*(n+b)*p2)/(temp*(1 - z*z))
            z1 = z
            z = z1 - p1/pp

            if abs(z - z1) < 1e-12:
                break

            its += 1

        if its == maxiter:
            raise ValueError("Max Iteration reached.  Failed to converge")

        nodes[i] = z
        weights[i] = temp/(pp*p2)

    nodes = (1-nodes)/2
    weights = weights * math.exp(gammaln(a+n) + gammaln(b+n) -
                                 gammaln(n+1) - gammaln(n+ab+1))
    weights = weights / (2*math.exp(gammaln(a+1) + gammaln(b+1) -
                         gammaln(ab+2)))

    return nodes, weights


@jit(nopython=True)
def _qnwgamma1(n, a=1.0, b=1.0, tol=3e-14):
    """
    1d quadrature weights and nodes for Gamma distributed random variable

    Parameters
    ----------
    n : scalar : int
        The number of quadrature points

    a : scalar : float, optional(default=1.0)
        Shape parameter of the gamma distribution parameter. Must be positive

    b : scalar : float, optional(default=1.0)
        Scale parameter of the gamma distribution parameter. Must be positive

    tol : scalar : float, optional(default=3e-14)
        Tolerance parameter for newton iterations for each node

    Returns
    -------
    nodes : np.ndarray(dtype=float, ndim=1)
        The quadrature points

    weights : np.ndarray(dtype=float, ndim=1)
        The quadrature weights that correspond to nodes

    Notes
    -----
    Based of original function ``qnwgamma1`` in CompEcon toolbox by
    Miranda and Fackler

    References
    ----------
    Miranda, Mario J, and Paul L Fackler. Applied Computational
    Economics and Finance, MIT Press, 2002.

    """
    a -= 1

    maxit = 25

    factor = -math.exp(gammaln(a+n) - gammaln(n) - gammaln(a+1))
    nodes = np.zeros(n)
    weights = np.zeros(n)

    # Create nodes
    for i in range(n):
        # Reasonable starting values
        if i == 0:
            z = (1+a) * (3+0.92*a) / (1 + 2.4*n + 1.8*a)
        elif i == 1:
            z = z + (15 + 6.25*a) / (1 + 0.9*a + 2.5*n)
        else:
            j = i-1
            z = z + ((1 + 2.55*j) / (1.9*j) + 1.26*j*a / (1 + 3.5*j)) * \
                (z - nodes[j-1]) / (1 + 0.3*a)

        # root finding iterations
        its = 0
        z1 = -10000
        while abs(z - z1) > tol and its < maxit:
            p1 = 1.0
            p2 = 0.0
            for j in range(1, n+1):
                # Recurrance relation for Laguerre polynomials
                p3 = p2
                p2 = p1
                p1 = ((2*j - 1 + a - z)*p2 - (j - 1 + a)*p3) / j

            pp = (n*p1 - (n+a)*p2) / z
            z1 = z
            z = z1 - p1/pp
            its += 1

        if its == maxit:
            raise ValueError('Failure to converge')

        nodes[i] = z
        weights[i] = factor / (pp*n*p2)

    return nodes*b, weights
