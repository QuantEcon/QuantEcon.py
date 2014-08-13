"""
Author: David R. Pugh

Solow (1956) model of economic growth focuses on the behavior of four variables:
output, `Y`, capital, `K`, labor, `L`, and knowledge (or technology or the
``effectiveness of labor''), `A`. At each point in time the economy has some
amounts of capital, labor, and knowledge that can be combined to produce output
according to some production function, `F`.

.. math::

    Y(t) = F(A(t), K(t), L(t))

where `t` denotes time.

"""
import numpy as np
import sympy as sp

# declare key variables for the model
A, k, K, L, t = sp.var('A, k, K, L, t')

# declare required model parameters
g, n, s, delta = sp.var('g, n, s, delta')


class Model(object):

    def __init__(self, output, params):
        self.output = output
        self.params = params

    @property
    def intensive_output(self):
        r"""
        The assumption of constant returns to scale allows us to work the the
        intensive form of the production function, `F`. Defining :math:`c=1/AL`
        one can write

        ..math::

            F\bigg(\frac{K}{AL}, 1) = \frac{1}{AL}F(A, K, L)

        Defining :math:`k=K/AL` and :math:`y=Y/AL` to be capital per effective
        worker and output per effective worker, respectively, the intensive
        form of the production function can be written as

        .. math::

            y = f(k).

        Tradionaly assumptions are that the function `f` satisfies :math:`f(0)=0`,
        is concave (i.e., :math:`f'(k) > 0, f''(k) < 0`), and satisfies the
        Inada (1964) conditions:

        .. math::

            \lim_{k \rigtharrow 0} = \infty \\
            \lim_{k \rightarrow \infty} = 0

        :getter: Return the current intensive production function.
        :type: sp.Basic

        """
        return self._intensive_output

    @property
    def output(self):
        r"""
        Solow (1956) model of economic growth focuses on the behavior of four
        variables: output, `Y`, capital, `K`, labor, `L`, and knowledge (or
        technology or the ``effectiveness of labor''), `A`. At each point in
        time the economy has some amounts of capital, labor, and knowledge that
        can be combined to produce output according to some function, `F`.

        .. math::

            Y(t) = F(K(t), A(t)L(t))

        where `t` denotes time. A key assumption of the Solow model is that the
        function `F` exhibits constant returns to scale in capital and labor.

        .. math::

            F(cK(t), cA(t)L(t)) = cF(K(t), A(t)L(t)) = cY(t)

        for any :math:`c \ge 0`.

        :getter: Return the current production function.
        :setter: Set a new production function
        :type: sp.Basic

        """
        return self._output

    @property
    def params(self):
        """
        Dictionary of model parameters.

        The following parameters are required:

        g : float
            Growth rate of technology.
        n : float
            Growth rate of the labor force.
        s : float
            Savings rate. Must satisfy ``0 < s < 1``.
        delta : float
            Depreciation rate of physical capital. Must satisfy
            :math:`0 < \delta`.

        Note that there will likely be additional model parameters specific to
        the specified production function.

        :getter: Return the current dictionary of model parameters.
        :setter: Set a new dictionary of model parameters.
        :type: dict

        """
        return self._params

    @output.setter
    def output(self, value):
        """Set a new production function."""
        self._output = self._validate_output(value)

        # set the intensive form
        self._intensive_output = self._output.subs({'A': 1.0, 'K': k, 'L': 1.0})

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

    def _validate_params(self, params):
        """Validate the model parameters."""
        if not isinstance(params, dict):
            mesg = "SolowModel.params must be a dict, not a {}."
            raise ValueError(mesg.format(params.__class__))
        if params['s'] <= 0.0 or params['s'] >= 1.0:
            raise ValueError('Savings rate must be in (0, 1).')
        if params['delta'] <= 0.0 or params['delta'] >= 1.0:
            raise ValueError('Depreciation rate must be in (0, 1).')
        if params['g'] + params['n'] + params['delta'] <= 0.0:
            raise ValueError("Sum of g, n, and delta must be positive.")
        else:
            return params


# define symbolic model equations
_k_dot = s * y - (g + n + delta) * k

# define symbolic system and compute the jacobian
X = sp.DeferredVector('X')
change_of_vars = {'k': X[0]}

_solow_system = sp.Matrix([_k_dot]).subs(change_of_vars)
_solow_jacobian = _solow_system.jacobian([X[0]])

# wrap the symbolic expressions as callable numpy funcs
_args = (t, X, g, n, s, alpha, delta, sigma)
_f = sp.lambdify(_args, _solow_system,
                 modules=[{'ImmutableMatrix': np.array}, "numpy"])
_jac = sp.lambdify(_args, _solow_jacobian,
                   modules=[{'ImmutableMatrix': np.array}, "numpy"])


def f(t, k, g, n, s, alpha, delta, sigma):
    """
    Equation of motion for capital (per worker/effective worker) for a
    Solow growth model with constant elasticity of substitution (CES)
    production function.

    Parameters
    ----------
    t : array_like (float)
        Time.
    X : ndarray (float, shape=(1,))
        Endogenous variables of the Solow model. Ordering is `X = [k]` where
        `k` is capital (per worker/effective worker).
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
    k_dot = _f(t, k, g, n, s, alpha, delta, sigma).ravel()
    return k_dot


def jacobian(t, X, g, n, s, alpha, delta, sigma):
    """
    Jacobian for the Solow model with constant elasticity of substitution (CES)
    production.

    Parameters
    ----------
    t : float
        Time.
    X : ndarray (float, shape=(1,))
        Endogenous variables of the Solow model. Ordering is `X = [k]` where
        `k` is capital (per worker/effective worker).
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
    jac = _jac(t, X, g, n, s, alpha, delta, sigma)
    return jac


def main():
    """Simple test case."""
    # define production function parameters
    alpha, sigma = sp.var('alpha, sigma')

    # define the the production function
    rho = (sigma - 1) / sigma
    Y = (alpha * K**rho + (1 - alpha) * (A * L)**rho)**(1 / rho)

    return Model(output=Y, params=None)

if __name__ == '__main__':
    model = main()
