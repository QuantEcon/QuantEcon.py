"""
Author: David R. Pugh

Solow (1956) model of economic growth.

"""
import numpy as np
import sympy as sp

# basic variables for a Solow model
A, k, K, L = sp.var('A, k, K, L')


class Model(object):

    def __init__(self, output, params):
        self.output = output
        self.params = params

    @property
    def output(self):
        """
        Aggregate production function.

        Output `Y` is assumed to be some function of technology, `A`, capital,
        `K`, and labor, `L`:

        .. math::

            Y = F(A, K, L).

        Standard assumptions are that the function `F` exhibits constant return
        to scale with respect to capital and labor inputs.

        :getter: Return the current production function.
        :setter: Set new production function
        :type: sp.Basic

        """
        return self._output

    @property
    def params(self):
        """
        Dictionary of model parameters.

        Parameters
        ----------
        g : float
            Growth rate of technology.
        n : float
            Growth rate of the labor force.
        s : float
            Savings rate. Must satisfy ``0 < s < 1``.
        delta : float
            Depreciation rate of physical capital. Must satisfy
            :math:`0 < \delta`.

        :getter: Return the current production function.
        :setter: Set new production function
        :type: sp.Basic

        """
        return self._params

    @output.setter
    def output(self, value):
        """Set a new production function."""
        self._output = self._validate_output(value)

    @params.setter
    def params(self, value):
        """Set a new parameter dictionary."""
        self._params = value

    def _validate_output(self, output):
        """Validate the production function."""
        if not isinstance(output, sp.Basic):
            mesg = ("Output must be an instance of {}.".format(sp.Basic))
            raise ValueError(mesg)
        if not ({A, K, L} < output.atoms()):
            mesg = ("Output must be an expression of technology, 'A', " +
                    "capital, 'K', and labor, 'L'.")
            raise ValueError(mesg)
        else:
            return output


# define symbolic model equations
#_k_dot = s * y - (g + n + delta) * k

# define symbolic system and compute the jacobian
#_solow_system = sp.Matrix([_k_dot])
#_solow_jacobian = _solow_system.jacobian([k])

# wrap the symbolic expressions as callable numpy funcs
#_args = (k, g, n, s, alpha, delta, sigma)
#_f = sp.lambdify(_args, _solow_system,
#                 modules=[{'ImmutableMatrix': np.array}, "numpy"])
#_jac = sp.lambdify(_args, _solow_jacobian,
#                   modules=[{'ImmutableMatrix': np.array}, "numpy"])


def f(t, k, g, n, s, alpha, delta, sigma):
    """
    Equation of motion for capital (per worker/effective worker) for a
    Solow growth model with constant elasticity of substitution (CES)
    production function.

    Parameters
    ----------
    t : array_like (float)
        Time.
    k : array_like (float)
        Capital (per worker/effective worker).
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
    k_dot = _f(k, g, n, s, alpha, delta, sigma).ravel()
    return k_dot


def jacobian(t, k, g, n, s, alpha, delta, sigma):
    """
    Jacobian for the Solow model with constant elasticity of substitution (CES)
    production.

    Parameters
    ----------
    t : array_like (float)
        Time.
    k : array_like (float)
        Capital (per worker/effective worker).
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
    jac = _jac(k, g, n, s, alpha, delta, sigma)
    return jac


def main():
    """Basic test case."""
    # declare model parameters
    g, n, s, alpha, delta, sigma = sp.var('g, n, s, alpha, delta, sigma')

    # define the intensive for for the production function
    rho = (sigma - 1) / sigma
    Y = (alpha * K**rho + (1 - alpha) * (A * L)**rho)**(1 / rho)

    return Model(output=Y, params=None)

if __name__ == '__main__':
    model = main()