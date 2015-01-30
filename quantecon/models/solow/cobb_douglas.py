"""
Solow growth model with Cobb-Douglas aggregate production.

@author : David R. Pugh
@date : 2014-11-27

"""
from __future__ import division
from textwrap import dedent

import numpy as np
import sympy as sym

from . import model

# declare key variables for the model
t, X = sym.symbols('t'), sym.DeferredVector('X')
A, k, K, L = sym.symbols('A, k, K, L')

# declare required model parameters
g, n, s, alpha, delta = sym.symbols('g, n, s, alpha, delta')


class CobbDouglasModel(model.Model):

    _required_params = ['g', 'n', 's', 'alpha', 'delta', 'A0', 'L0']

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

    def __str__(self):
        """Human readable summary of a CESModel instance."""
        m = super(CobbDouglasModel, self).__str__()
        m += "  - alpha (output elasticity)                     : {alpha:g}\n"
        formatted_str = dedent(m.format(alpha=self.params['alpha']))
        return formatted_str

    @property
    def steady_state(self):
        r"""
        Steady state value of capital stock (per unit effective labor).

        :getter: Return the current steady state value.
        :type: float

        Notes
        -----
        The steady state value of capital stock (per unit effective labor)
        with Cobb-Douglas production is defined as

        .. math::

            k^* = \bigg(\frac{s}{g + n + \delta}\bigg)^\frac{1}{1-\alpha}

        where `s` is the savings rate, :math:`g + n + \delta` is the effective
        depreciation rate, and :math:`\alpha` is the elasticity of output with
        respect to capital (i.e., capital's share).

        """
        s = self.params['s']
        alpha = self.params['alpha']
        return (s / self.effective_depreciation_rate)**(1 / (1 - alpha))

    def _validate_params(self, params):
        """Validate the model parameters."""
        params = super(CobbDouglasModel, self)._validate_params(params)
        if params['alpha'] <= 0.0 or params['alpha'] >= 1.0:
            raise AttributeError('Output elasticity must be in (0, 1).')
        else:
            return params

    def analytic_solution(self, t, k0):
        """
        Compute the analytic solution for the Solow model with Cobb-Douglas
        production technology.

        Parameters
        ----------
        t : numpy.ndarray (shape=(T,))
            Array of points at which the solution is desired.
        k0 : (float)
            Initial condition for capital stock (per unit of effective labor)

        Returns
        -------
        analytic_traj : ndarray (shape=t.size, 2)
            Array representing the analytic solution trajectory.

        """
        s = self.params['s']
        alpha = self.params['alpha']

        # lambda governs the speed of convergence
        lmbda = self.effective_depreciation_rate * (1 - alpha)

        # analytic solution for Solow model at time t
        k_t = (((s / (self.effective_depreciation_rate)) * (1 - np.exp(-lmbda * t)) +
                k0**(1 - alpha) * np.exp(-lmbda * t))**(1 / (1 - alpha)))

        # combine into a (T, 2) array
        analytic_traj = np.hstack((t[:, np.newaxis], k_t[:, np.newaxis]))

        return analytic_traj
