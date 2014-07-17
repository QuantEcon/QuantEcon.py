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
        MPK = np.nan * K
    else:
        MPK = ((alpha * K**(rho - 1) / (alpha * K**rho + beta * (A * L)**rho)) *
               output(K, A, L, alpha, sigma))

    return MPK


def output(K, A, L, alpha, beta, sigma):
    """
    Constant elasticity of substitution (CES) production function with labor
    augmenting technology.

    Arguments
    ---------
    K : array_like (float)
        Capital
    L : array_like (float)
        Labor
    A : array_like (float)
        Technology
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
    