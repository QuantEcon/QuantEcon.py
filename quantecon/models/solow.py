"""
Author: David R. Pugh

Solow (1956) model of economic growth.

"""
import numpy as np
import sympy as sp

# declare endogenous variables
k = sp.var('k')

# declare model parameters
g, n, s, alpha, delta, sigma = sp.var('g, n, s, alpha, delta, sigma')

# define the intensive for for the production function
rho = (sigma - 1) / sigma
y = (alpha * k**rho + (1 - alpha))**(1 / rho)

# define symbolic model equations
_k_dot = s * y - (g + n + delta) * k

# define symbolic system and compute the jacobian
_solow_system = sp.Matrix([_k_dot])
_solow_jacobian = _solow_system.jacobian([k])

# wrap the symbolic expressions as callable numpy funcs
_args = (k, g, n, s, alpha, delta, sigma)
_f = sp.lambdify(_args, _solow_system,
                 modules=[{'ImmutableMatrix': np.array}, "numpy"])
_jac = sp.lambdify(_args, _solow_jacobian,
                   modules=[{'ImmutableMatrix': np.array}, "numpy"])


def f(t, k, g, n, s, alpha, delta, sigma):
    """
    Equation of motion for capital (per worker/effective worker) for a
    Solow growth model with constant elasticity of substitution (CES)
    production function.

    Parameters
    ----------
    t : array_like (float)
        Time.
    k : array_like (float)
        Capital (per worker/effective worker).
    g : float
        Growth rate of technology.
    n : float
        Growth rate of the labor force.
    s : float
        Savings rate. Must satisfy ``0 < s < 1``.
    alpha : float
        Importance of capital relative to effective labor in production. Must
        satisfy :math:`0 < \alpha < 1`.
    delta : float
        Depreciation rate of physical capital. Must satisfy
        :math:`0 < \delta`.
    sigma : float
        Elasticity of substitution between capital and effective labor in
        production. Must satisfy :math:`0 \le \sigma`.

    Returns
    -------
    k_dot : array_like (float)
        Rate of change of capital (per worker/effective worker).

    """
    k_dot = _f(k, g, n, s, alpha, delta, sigma).ravel()
    return k_dot


def jacobian(t, k, g, n, s, alpha, delta, sigma):
    """
    Jacobian for the Solow model with constant elasticity of substitution (CES)
    production.

    Parameters
    ----------
    t : array_like (float)
        Time.
    k : array_like (float)
        Capital (per worker/effective worker).
    g : float
        Growth rate of technology.
    n : float
        Growth rate of the labor force.
    s : float
        Savings rate. Must satisfy ``0 < s < 1``.
    alpha : float
        Importance of capital relative to effective labor in production. Must
        satisfy :math:`0 < \alpha < 1`.
    delta : float
        Depreciation rate of physical capital. Must satisfy
        :math:`0 < \delta`.
    sigma : float
        Elasticity of substitution between capital and effective labor in
        production. Must satisfy :math:`0 \le \sigma`.

    Returns
    -------
    jac : array_like (float)
        Derivative of the equation of motion for capital (per worker/effective
        worker) with respect to `k`.

    """
    jac = _jac(k, g, n, s, alpha, delta, sigma)
    return jac
