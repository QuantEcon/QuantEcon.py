"""
Solow model with constant elasticity of substitution (CES) production.

@author : David R. Pugh
@date : 2014-11-29

TODO:

Implement additional check on parameters that insures a non-infinite steady state.

"""
import sympy as sym

from . import model

# declare key variables for the model
t, X = sym.symbols('t'), sym.DeferredVector('X')
A, k, K, L = sym.symbols('A, k, K, L')

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

    def _isfinite_steady_state(self, params):
        """Check whether parameters are consistent with finite steady state."""
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
        if params['alpha'] <= 0.0 or params['alpha'] >= 1.0:
            raise AttributeError('Output elasticity must be in (0, 1).')
        elif params['alpha'] <= 0.0:
            mesg = 'Elasticity of substitution must be strictly positive.'
            raise AttributeError(mesg)
        elif not self._isfinite_steady_state(params):
            mesg = 'Parameters are inconsistent with finite steady state.'
            raise AttributeError(mesg)
        else:
            return params
