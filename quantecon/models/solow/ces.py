"""
Solow model with constant elasticity of substitution (CES) production.

"""
import numpy as np
import sympy as sym

from . import model

# declare key variables for the model
t, X = sym.symbols('t'), sym.DeferredVector('X')
A, k, K, L = sym.symbols('A, k, K, L')

# declare required model parameters
g, n, s, alpha, delta, sigma = sym.symbols('g, n, s, alpha, delta, sigma')


class CESModel(model.Model):

    def __init__(self, params):
        """
        Create an instance of the Solow growth model with constant elasticity
        of subsitution (CES) aggregate production.

        Parameters
        ----------
        params : dict
            Dictionary of model parameters.

        """
        rho = (sigma - 1) / sigma
        ces_output = (alpha * K**rho + (1 - alpha) * (A * L)**rho)**(1 / rho)
        super(CESModel, self).__init__(ces_output, params)


def _cobb_douglas_steady_state(g, n, s, alpha, delta):
    """
    Steady-state level of capital stock (per unit effective labor) for a
    Solow growth model with Cobb-Douglas aggregate production.

    """
    k_star = (s / (n + g + delta))**(1 / (1 - alpha))
    return k_star


def _leontief_steady_state(g, n, s, alpha, delta):
    """
    Steady-state level of capital stock (per unit effective labor) for a
    Solow growth model with leontief aggregate production.

    """
    raise NotImplementedError


def _general_ces_steady_state(g, n, s, alpha, delta, sigma):
    """
    Steady-state level of capital stock (per unit effective labor) for a
    Solow growth model with CES aggregate production.

    """
    rho = (sigma - 1) / sigma
    k_star = ((1 / (1 - alpha)) * ((s / (g + n + delta))**-rho - alpha))**(-1 / rho)
    return k_star


def ces_steady_state(g, n, s, alpha, delta, sigma):
    """
    Steady-state level of capital stock (per unit effective labor) for a
    Solow growth model with constant elasticity of substitution (CES) aggregate
    production.

    Parameters
    ----------
    g : float
        Growth rate of technology.
    n : float
        Growth rate of the labor force.
    s : float
        Savings rate. Must satisfy `0 < s < 1`.
    alpha : float
        Importance of capital stock relative to effective labor in the
        production of output. Constant returns to scale requires that
        :math:`0 < alpha < 1`.
    delta : float
        Depreciation rate of physical capital. Must satisfy :math:`0 < \delta`.
    sigma : float
        Elasticity of substitution between capital stock and effective labor in
        the production of output.

    Returns
    -------
    k_star : float
        Steady state value for capital stock (per unit of effective labor).

    """
    if np.isclose(sigma, 0.0):
        k_star = _leontief_steady_state(g, n, s, alpha, delta)
    elif np.isclose(sigma, 1.0):
        k_star = _cobb_douglas_steady_state(g, n, s, alpha, delta)
    else:
        k_star = _general_ces_steady_state(g, n, s, alpha, delta, sigma)

    return k_star
