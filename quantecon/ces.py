from __future__ import division

import numpy as np


def marginal_product_capital(K, A, L, alpha, beta, sigma):
    """
    Marginal product of capital for constant elasticity of substitution (CES)
    production function with labor augmenting technology.

    Parameters
    ----------
    K : array_like (float)
        Capital
    A : array_like (float)
        Technology
    L : array_like (float)
        Labor
    alpha : float
        Importance of capital in production. Must satisfy :math:`0 < \alpha`.
    beta : float
        Importance of effective labor in production. Must satisfy
        :math:`0 < \beta`.
    sigma : float
        Elasticity of substitution between capital and effective labor in
        production. Must satisfy :math:`0 \le \sigma`.

    Returns
    -------
    MPK : array_like (float)
        Marginal product of capital.

    """
    # CES nests both Cobb-Douglas and Leontief functions
    if np.isclose(sigma, 1.0):
        MPK = alpha * K**(alpha - 1) * (A * L)**beta
    elif np.isclose(sigma, 0.0):
        MPK = np.where(alpha * K < beta * A * L, alpha, 0)
    else:
        rho = (sigma - 1) / sigma
        MPK = ((alpha * K**(rho - 1) / (alpha * K**rho + beta * (A * L)**rho)) *
               output(K, A, L, alpha, beta, sigma))

    return MPK


def marginal_product_labor(K, A, L, alpha, beta, sigma):
    """
    Marginal product of labor for constant elasticity of substitution (CES)
    production function with labor augmenting technology.

    Parameters
    ----------
    K : array_like (float)
        Capital
    A : array_like (float)
        Technology
    L : array_like (float)
        Labor
    alpha : float
        Importance of capital in production. Must satisfy :math:`0 < \alpha`.
    beta : float
        Importance of effective labor in production. Must satisfy
        :math:`0 < \beta`.
    sigma : float
        Elasticity of substitution between capital and effective labor in
        production. Must satisfy :math:`0 \le \sigma`.

    Returns
    -------
    MPL : array_like (float)
        Marginal product of labor.

    """
    # CES nests both Cobb-Douglas and Leontief functions
    if np.isclose(sigma, 1.0):
        MPL = beta * K**alpha * (A * L)**(beta - 1) * A
    elif np.isclose(sigma, 0.0):
        MPL = np.where(beta * A * L < alpha * K, beta * A, 0)
    else:
        rho = (sigma - 1) / sigma
        MPL = ((beta * (A * L)**(rho - 1) * A / (alpha * K**rho + beta * (A * L)**rho)) *
               output(K, A, L, alpha, beta, sigma))

    return MPL


def output(K, A, L, alpha, beta, sigma):
    """
    Constant elasticity of substitution (CES) production function with labor
    augmenting technology.

    Parameters
    ----------
    K : array_like (float)
        Capital
    A : array_like (float)
        Technology
    L : array_like (float)
        Labor
    alpha : float
        Importance of capital in production. Must satisfy :math:`0 < \alpha`.
    beta : float
        Importance of effective labor in production. Must satisfy
        :math:`0 < \beta`.
    sigma : float
        Elasticity of substitution between capital and effective labor in
        production. Must satisfy :math:`0 \le \sigma`.

    Returns
    -------
    Y : array_like (float)
        Output

    """
    # CES nests both Cobb-Douglas and Leontief functions
    if np.isclose(sigma, 1.0):
        Y = K**alpha * (A * L)**beta
    elif np.isclose(sigma, 0.0):
        Y = np.minimum(alpha * K, beta * A * L)
    else:
        rho = (sigma - 1) / sigma
        Y = (alpha * K**rho + beta * (A * L)**rho)**(1 / rho)

    return Y


def output_elasticity_capital(K, A, L, alpha, beta, sigma):
    """
    Elasticity of output with respect to capital for the constant elasticity
    of substitution (CES) production function with labor augmenting technology.

    Parameters
    ----------
    K : array_like (float)
        Capital
    A : array_like (float)
        Technology
    L : array_like (float)
        Labor
    alpha : float
        Importance of capital in production. Must satisfy :math:`0 < \alpha`.
    beta : float
        Importance of effective labor in production. Must satisfy
        :math:`0 < \beta`.
    sigma : float
        Elasticity of substitution between capital and effective labor in
        production. Must satisfy :math:`0 \le \sigma`.

    Returns
    -------
    epsilon_YK : array_like (float)
        Output elasticity with respect to capital.

    """
    Y = output(K, A, L, alpha, beta, sigma)
    epsilon_YK = (K / Y) * marginal_product_capital(K, A, L, alpha, beta, sigma)
    return epsilon_YK


def output_elasticity_labor(K, A, L, alpha, beta, sigma):
    """
    Elasticity of output with respect to labor for the constant elasticity
    of substitution (CES) production function with labor augmenting technology.

    Parameters
    ----------
    K : array_like (float)
        Capital
    A : array_like (float)
        Technology
    L : array_like (float)
        Labor
    alpha : float
        Importance of capital in production. Must satisfy :math:`0 < \alpha`.
    beta : float
        Importance of effective labor in production. Must satisfy
        :math:`0 < \beta`.
    sigma : float
        Elasticity of substitution between capital and effective labor in
        production. Must satisfy :math:`0 \le \sigma`.

    Returns
    -------
    epsilon_YL : array_like (float)
        Output elasticity with respect to labor.

    """
    Y = output(K, A, L, alpha, beta, sigma)
    epsilon_YL = (L / Y) * marginal_product_labor(K, A, L, alpha, beta, sigma)
    return epsilon_YL