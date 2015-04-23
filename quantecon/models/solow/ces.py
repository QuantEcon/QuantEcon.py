"""
Solow model with constant elasticity of substitution (CES) production.

@author : David R. Pugh
@date : 2014-12-11

"""
from __future__ import division
from textwrap import dedent

import sympy as sym

from . import model

# declare key variables for the model
A, k, K, L, Y = sym.symbols('A, k, K, L, Y')

# declare required model parameters
g, n, s, alpha, delta, sigma = sym.symbols('g, n, s, alpha, delta, sigma')


class CESModel(model.Model):

    _required_params = ['g', 'n', 's', 'alpha', 'delta', 'sigma', 'A0', 'L0']

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

    def __str__(self):
        """Human readable summary of a CESModel instance."""
        m = super(CESModel, self).__str__()
        m += "  - alpha (capital's weight in output)            : {alpha:g}\n"
        m += "  - sigma (elasticity of substitution)            : {sigma:g}"
        formatted_str = dedent(m.format(alpha=self.params['alpha'],
                                        sigma=self.params['sigma']))
        return formatted_str

    @property
    def solow_residual(self):
        """
        Symbolic expression for the Solow residual which is used as a measure
        of technology.

        :getter: Return the symbolic expression.
        :type: sym.Basic

        """
        rho = (sigma - 1) / sigma
        residual = (((1 / (1 - alpha)) * (Y / L)**rho -
                     (alpha / (1 - alpha)) * (K / L)**rho)**(1 / rho))
        return residual

    @property
    def steady_state(self):
        r"""
        Steady state value of capital stock (per unit effective labor).

        :getter: Return the current steady state value.
        :type: float

        Notes
        -----
        The steady state value of capital stock (per unit effective labor)
        with CES production is defined as

        .. math::

            k^* = \left[\frac{1-\alpha}{\bigg(\frac{g+n+\delta}{s}\bigg)^{\rho}-\alpha}\right]^{\frac{1}{rho}}

        where `s` is the savings rate, :math:`g + n + \delta` is the effective
        depreciation rate, and :math:`\alpha` controls the importance of
        capital stock relative to effective labor in the production of output.
        Finally,

        ..math::

            \rho=\frac{\sigma-1}{\sigma}

        where `:math:`sigma` is the elasticity of substitution between capital
        and effective labor in production.

        """
        g = self.params['g']
        n = self.params['n']
        s = self.params['s']
        alpha = self.params['alpha']
        delta = self.params['delta']
        sigma = self.params['sigma']

        ratio_investment_rates = (g + n + delta) / s
        rho = (sigma - 1) / sigma
        k_star = ((1 - alpha) / (ratio_investment_rates**rho - alpha))**(1 / rho)

        return k_star

    def _isdeterminate_steady_state(self, params):
        """Check that parameters are consistent with determinate steady state."""
        g = params['g']
        n = params['n']
        s = params['s']
        alpha = params['alpha']
        delta = params['delta']
        sigma = params['sigma']

        ratio_investment_rates = (g + n + delta) / s
        rho = (sigma - 1) / sigma

        return ratio_investment_rates**rho - alpha > 0

    def _validate_params(self, params):
        """Validate the model parameters."""
        params = super(CESModel, self)._validate_params(params)
        if params['alpha'] < 0.0 or params['alpha'] > 1.0:
            raise AttributeError('Capital weight must be in (0, 1).')
        elif params['sigma'] <= 0.0:
            mesg = 'Elasticity of substitution must be strictly positive.'
            raise AttributeError(mesg)
        elif not self._isdeterminate_steady_state(params):
            mesg = 'Steady state is indeterminate.'
            raise AttributeError(mesg)
        else:
            return params
