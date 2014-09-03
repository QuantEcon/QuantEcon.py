"""
Solow growth model with Cobb-Douglas aggregate production.

@author : David R. Pugh
@date : 2014-09-01

"""
import numpy as np
import sympy as sym

from . import model

# declare key variables for the model
t, X = sym.var('t'), sym.DeferredVector('X')
A, k, K, L = sym.var('A, k, K, L')

# declare required model parameters
g, n, s, alpha, delta = sym.var('g, n, s, alpha, delta')


class CobbDouglasModel(model.Model):

    def __init__(self, params):
        """
        Create an instance of the Solow growth model with Cobb-Douglas
        aggregate production.

        Parameters
        ----------
        params : dict
            Dictionary of model parameters.

        """
        cobb_douglas_output = K**alpha * (A * L)**(1 - alpha)
        super(CobbDouglasModel, self).__init__(cobb_douglas_output, params)


def analytic_solution(t, k0, g, n, s, alpha, delta):
    """
    Compute the analytic solution for the Solow model with Cobb-Douglas
    production technology.

    Parameters
    ----------
    t : ndarray (shape=(T,))
        Array of points at which the solution is desired.
    k0 : (float)
        Initial condition for capital stock (per unit of effective labor)
    g : float
        Growth rate of technology.
    n : float
        Growth rate of the labor force.
    s : float
        Savings rate. Must satisfy `0 < s < 1`.
    alpha : float
        Elasticity of output with respect to capital stock.
    delta : float
        Depreciation rate of physical capital. Must satisfy
        :math:`0 < \delta`.

    Returns
    -------
    analytic_traj : ndarray (shape=t.size, 2)
        Array representing the analytic solution trajectory.

    """
    # lambda governs the speed of convergence
    lmbda = (n + g + delta) * (1 - alpha)

    # analytic solution for Solow model at time t
    k_t = (((s / (n + g + delta)) * (1 - np.exp(-lmbda * t)) +
            k0**(1 - alpha) * np.exp(-lmbda * t))**(1 / (1 - alpha)))

    # combine into a (T, 2) array
    analytic_traj = np.hstack((t[:, np.newaxis], k_t[:, np.newaxis]))

    return analytic_traj


def analytic_steady_state(g, n, s, alpha, delta):
    """
    Steady-state level of capital stock (per unit effective labor).

    Parameters
    ----------
    g : float
        Growth rate of technology.
    n : float
        Growth rate of the labor force.
    s : float
        Savings rate. Must satisfy `0 < s < 1`.
    alpha : float
        Elasticity of output with respect to capital stock. Must satisfy
        :math:`0 < alpha < 1`.
    delta : float
        Depreciation rate of physical capital. Must satisfy :math:`0 < \delta`.

    Returns
    -------
    kstar : float
        Steady state value of capital stock (per unit effective labor).

    """
    k_star = (s / (n + g + delta))**(1 / (1 - alpha))
    return k_star
