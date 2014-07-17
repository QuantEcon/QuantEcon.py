from __future__ import division

import numpy as np

import ces
import ivp


class Model(ivp.IVP):
    """Base class representing a continuous time Solow growth model."""

    def __init__(self, k_dot, jacobian, params):
        """
        Creates an instance of the Solow model.

        Arguments
        ----------
        k_dot : callable, ``k_dot(t, k, *params)``
            Equation of motion for capital (per worker/effective worker). The
            independent variable, `t`, is time; `k`, is capital (per worker/
            effective worker); `params` is a tuple of model parameters.
        jacobian : callable, ``jacobian(t, k, *params)``
            The derivative of the equation of motion for capital (per worker/
            effective worker) with respect to `k`. The independent variable, t,
            is time; k, (per worker/effective worker); `params` is a tuple
            of model parameters.
        params : tuple
            Tuple of model parameters. Standard parameters for a Solow
            growth model are:

            - `g`: Growth rate of technology (rate of technological progress).
            - `n`: Growth rate of labor force.
            - `s`: Savings rate. Must satisfy ``0 < s < 1``.
            - :math:`\delta`: Depreciation rate of physical capital. Must
            satisfy :math:`0 < \delta`.

            Only other model parameters will be the parameters of some
            production function.

        """
        super(Model, self).__init__(k_dot, jacobian, params)


def ces_actual_investment(k, s, alpha, sigma):
    """
    Total amount of output (per worker/effective worker) invested into the
    production of new capital.

    Arguments
    ---------
    k : array_like (float)
        Capital (per worker/effective worker).
    s : float
        Savings rate. Must satisfy ``0 < s < 1``.
    alpha : float
        Importance of capital relative to effective labor in production. Must
        satisfy :math:`0 < \alpha < 1`.
    sigma : float
        Elasticity of substitution between capital and effective labor in
        production. Must satisfy :math:`0 \le \sigma`.

    Returns
    -------
    actual_investment : array_like (float)
        Total amount of output (per worker/effective worker) invested into the
        production of new capital.

    """
    actual_investment = s * ces.output(k, 1, 1, alpha, 1-alpha, sigma)
    return actual_investment


def ces_break_even_investment(k, g, n, delta):
    """
    Amount of investment required to maintain current levels of capital (per
    worker/effective worker).

    Arguments
    ---------
    k : array_like (float)
        Capital (per worker/effective worker).
    g : float
        Growth rate of technology.
    n : float
        Growth rate of the labor force.
    delta : float
        Depreciation rate of physical capital. Must satisfy
        :math:`0 < \delta`.

    Returns
    -------
    break_even_investment : array_like (float)
        Amount of investment required to maintain current levels of capital
        (per worker/effective worker).

    """
    break_even_investment = (g + n + delta) * k
    return break_even_investment


def ces_jacobian(k, t, g, n, s, alpha, delta, sigma):
    """
    Jacobian for the Solow model with constant elasticity of substitution (CES)
    production.

    Arguments
    ---------
    k : array_like (float)
        Capital (per worker/effective worker).
    t : array_like (float)
        Time.
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
    jac = (s * ces.marginal_product_capital(k, 1, 1, alpha, 1-alpha, sigma) -
           (g + n + delta))
    return jac


def ces_k_dot(k, t, g, n, s, alpha, delta, sigma):
    """
    Equation of motion for capital (per worker/effective worker) for a
    Solow growth model with constant elasticity of substitution (CES)
    production function.

    Arguments
    ---------
    k : array_like (float)
        Capital (per worker/effective worker).
    t : array_like (float)
        Time.
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
    k_dot = (ces_actual_investment(k, s, alpha, sigma) -
             ces_break_even_investment(k, g, n, delta))
    return k_dot


def cobb_douglas_analytic_solution(k0, t, g, n, s, alpha, delta):
    """
    Analytic solution for the Solow model with Cobb-Douglas production.

    Arguments
    ---------
        k0 : float
            Initial value for capital (per worker/effective worker)
        t : array_like (float, shape=(T,))
            Array of points at which the solution is desired.
        g : float
            Growth rate of technology.
        n : float
            Growth rate of the labor force.
        s : float
            Savings rate. Must satisfy ``0 < s < 1``.
        alpha : float
            Elasticity of output with respect to capital. Must satisfy
            :math:`0 < \alpha < 1`.
        delta : float
            Depreciation rate of physical capital. Must satisfy
            :math:`0 < \delta`.

    Returns
    -------
        analytic_traj : array_like (float, shape=(T,2))
            Array representing the analytic solution trajectory.

    """
    # speed of convergence
    lmbda = (n + g + delta) * (1 - alpha)

    # analytic solution for Solow model at time t
    k_t = (((s / (n + g + delta)) * (1 - np.exp(-lmbda * t)) +
           k0**(1 - alpha) * np.exp(-lmbda * t))**(1 / (1 - alpha)))

    # combine into a (T, 2) array
    analytic_traj = np.hstack((t[:, np.newaxis], k_t[:, np.newaxis]))

    return analytic_traj
