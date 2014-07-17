from __future__ import division

import numpy as np


def marginal_product_capital(K, A, L, alpha, beta, sigma):
    """
    Marginal product of capital for constant elasticity of substitution (CES)
    production function with labor augmenting technology.

    Arguments
    ---------
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
    rho = (sigma - 1) / sigma

    # CES nests both Cobb-Douglas and Leontief functions
    if abs(rho) < 1e-3:
        MPK = alpha * K**(alpha - 1) * (A * L)**beta
    elif sigma < 1e-3:
        raise NotImplementedError
    else:
        MPK = ((1 / K) * output_elasticity_capital(K, A, L, alpha, beta, sigma) *
               output(K, A, L, alpha, beta, sigma))

    return MPK


def marginal_product_labor(K, A, L, alpha, beta, sigma):
    """
    Marginal product of labor for constant elasticity of substitution (CES)
    production function with labor augmenting technology.

    Arguments
    ---------
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
    rho = (sigma - 1) / sigma

    # CES nests both Cobb-Douglas and Leontief functions
    if abs(rho) < 1e-3:
        MPL = (1 - beta) * K**alpha * (A * L)**(beta - 1)
    elif sigma < 1e-3:
        raise NotImplementedError
    else:
        MPL = ((1 / L) * output_elasticity_labor(K, A, L, alpha, beta, sigma) *
               output(K, A, L, alpha, beta, sigma))

    return MPL


def output(K, A, L, alpha, beta, sigma):
    """
    Constant elasticity of substitution (CES) production function with labor
    augmenting technology.

    Arguments
    ---------
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
    rho = (sigma - 1) / sigma

    # CES nests both Cobb-Douglas and Leontief functions
    if abs(rho) < 1e-3:
        Y = K**alpha * (A * L)**beta
    elif sigma < 1e-3:
        Y = np.minimum(alpha * K, beta * A * L)
    else:
        Y = (alpha * K**rho + beta * (A * L)**rho)**(1 / rho)

    return Y


def output_elasticity_capital(K, A, L, alpha, beta, sigma):
    """
    Elasticity of output with respect to capital for the constant elasticity
    of substitution (CES) production function with labor augmenting technology.

    Arguments
    ---------
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
    rho = (sigma - 1) / sigma

    # CES nests both Cobb-Douglas and Leontief functions
    if abs(rho) < 1e-3:
        epsilon_YK = alpha
    elif sigma < 1e-3:
        raise NotImplementedError
    else:
        epsilon_YK = alpha * K**rho / (alpha * K**rho + beta * (A * L)**rho)

    return epsilon_YK


def output_elasticity_labor(K, A, L, alpha, beta, sigma):
    """
    Elasticity of output with respect to labor for the constant elasticity
    of substitution (CES) production function with labor augmenting technology.

    Arguments
    ---------
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
    rho = (sigma - 1) / sigma

    # CES nests both Cobb-Douglas and Leontief functions
    if abs(rho) < 1e-3:
        epsilon_YL = beta
    elif sigma < 1e-3:
        raise NotImplementedError
    else:
        epsilon_YL = beta * (A * L)**rho / (alpha * K**rho + beta * (A * L)**rho)

    return epsilon_YL