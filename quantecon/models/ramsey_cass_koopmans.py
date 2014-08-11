"""
Author: David R. Pugh

Ramsey-Cass-Koopmans optimal growth model.

"""
import numpy as np
import sympy as sp

# declare independent and endogenous variables
t, c, k = sp.var('c, k')

# declare model parameters
g, n = sp.var('g, n')

# declare household parameters
rho, theta, delta = sp.var('rho, theta, delta')

# declare firm parameters
alpha, sigma = sp.var('alpha, sigma')

# define the intensive form of the production function
tmp_rho = (sigma - 1) / sigma
y = (alpha * k**tmp_rho + (1 - alpha))**(1 / tmp_rho)

# compute the marginal product of capital
mpk = sp.diff(y, k)

# consumption Euler equation
_c_dot = ((mpk - rho - theta * g) / theta) * c

# equation of motion for capital (per worker/effective worker)
_k_dot = y - c - (g + n + delta) * k

# define symbolic system and compute the jacobian
X = sp.DeferredVector('X')
change_of_vars = {'c': X[0], 'k': X[1]}

_ramsey_system = sp.Matrix([_c_dot, _k_dot]).subs(change_of_vars)
_ramsey_jacobian = _ramsey_system.jacobian([X[0], X[1]])

# wrap the symbolic expressions as callable numpy funcs
_args = (t, X, g, n, alpha, delta, rho, sigma, theta)
_f = sp.lambdify(_args, _ramsey_system,
                 modules=[{'ImmutableMatrix': np.array}, "numpy"])
_jac = sp.lambdify(_args, _ramsey_jacobian,
                   modules=[{'ImmutableMatrix': np.array}, "numpy"])


def f(t, X, g, n, alpha, delta, rho, sigma, theta):
    """
    Equation of motion for capital (per worker/effective worker) for a
    Ramsey-Cass-Koopmans optimal growth model with constant elasticity
    of substitution (CES) production function and CES household
    preferences.

    Parameters
    ----------
    t : array_like (float)
        Time.
    X : array_like (float, shape=(2,))
        Endogenous variables of the Ramsey-Cass-Koopmans model. Ordering
        is X = [c, k], where c is consumption (per worker/effective worker)
        and k is capital (per worker/effective worker).
    g : float
        Growth rate of technology.
    n : float
        Growth rate of the labor force.
    alpha : float
        Importance of capital relative to effective labor in production. Must
        satisfy :math:`0 < \alpha < 1`.
    delta : float
        Depreciation rate of physical capital. Must satisfy
        :math:`0 < \delta`.
    rho : float
        Discount rate of the representative household. Must satisfy
        :math:`0 \le rho`.
    sigma : float
        Elasticity of substitution between capital and effective labor in
        production. Must satisfy :math:`0 \le \sigma`.
    theta : float
        Coefficient of relative risk aversion (or inverse of the inter-temporal
        elasticity of substituion).

    Returns
    -------
    rhs_ode : array_like (float, shape=(2,))
        Right hand side of the ODE describing the Ramsey-Cass-Koopmans model.

    """
    rhs_ode = _f(t, X, g, n, alpha, delta, rho, sigma, theta).ravel()
    return rhs_ode


def jacobian(t, X, g, n, alpha, delta, rho, sigma, theta):
    """
    Jacobian for the Ramsey-Cass-Koopmans model.

    Parameters
    ----------
    t : array_like (float)
        Time.
    X : array_like (float, shape=(2,))
        Endogenous variables of the Ramsey-Cass-Koopmans model. Ordering
        is X = [c, k], where c is consumption (per worker/effective worker)
        and k is capital (per worker/effective worker).
    g : float
        Growth rate of technology.
    n : float
        Growth rate of the labor force.
    alpha : float
        Importance of capital relative to effective labor in production. Must
        satisfy :math:`0 < \alpha < 1`.
    delta : float
        Depreciation rate of physical capital. Must satisfy
        :math:`0 < \delta`.
    rho : float
        Discount rate of the representative household. Must satisfy
        :math:`0 \le rho`.
    sigma : float
        Elasticity of substitution between capital and effective labor in
        production. Must satisfy :math:`0 \le \sigma`.
    theta : float
        Coefficient of relative risk aversion (or inverse of the inter-temporal
        elasticity of substituion).

    Returns
    -------
    jac : array_like (float, shape=(2,2))
        Jacobian matrix of partial derivatives.

    """
    jac = _jac(t, X, g, n, alpha, delta, rho, sigma, theta)
    return jac
