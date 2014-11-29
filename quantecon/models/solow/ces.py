"""
Solow model with constant elasticity of substitution (CES) production.

@author : David R. Pugh
@date : 2014-11-29

"""
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

            k^* = \bigg[\bigg(\frac{1}{1 - alpha}\bigg)\bigg(\frac{s}{g + n + delta}\bigg)^{-rho} - alpha\bigg)\bigg]^{-\frac{1}{rho}}

        where `s` is the savings rate, :math:`g + n + \delta` is the effective
        depreciation rate, and :math:`\alpha` controls the importance of
        capital stock relative to effective labor in the production of output.
        Finally,

        ..math::

            \rho=\frac{\sigma -1}{\sigma}

        where `:math:`sigma` is the elasticity of substitution between capital
        and effective labor in production.

        """
        g = self.params['g']
        n = self.params['n']
        s = self.params['s']
        alpha = self.params['alpha']
        delta = self.params['delta']
        sigma = self.params['sigma']

        rho = (sigma - 1) / sigma
        k_star = ((1 / (1 - alpha)) * ((s / (g + n + delta))**-rho - alpha))**(-1 / rho)
        return k_star
