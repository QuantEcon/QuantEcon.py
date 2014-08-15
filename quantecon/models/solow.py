r"""
Author: David R. Pugh

The [solow1956] model of economic growth focuses on the behavior of four
variables: output, `Y`, capital, `K`, labor, `L`, and knowledge (or technology
or the ``effectiveness of labor''), `A`. At each point in time the economy has
some amounts of capital, labor, and knowledge that can be combined to produce
output according to some production function, `F`.

.. math::

    Y(t) = F(K(t), A(t)L(t))

where `t` denotes time.

References
----------
.. [romer2011] D. Romer. *Advanced Macroeconomics, 4th edition*, MacGraw Hill,
2011.
.. [solow1956] R. Solow. *A contribution to the theory of economic growth*,
Quarterly Journal of Economics, 70(1):64-95, 1956.

TODO:

1. Write a short demo notebook
2. Have properties return callable funcs for plotting.
3. Finish writing docs

"""
import numpy as np
import sympy as sp

from .. import ivp


# declare key variables for the model
t, X = sp.var('t'), sp.DeferredVector('X')
A, k, K, L = sp.var('A, k, K, L')

# declare required model parameters
g, n, s, delta = sp.var('g, n, s, delta')


class Model(ivp.IVP):

    def __init__(self, output, params):
        """
        Create an instance of the Solow growth model.

        Parameters
        ----------
        output : sp.Basic
            Symbolic expression defining the aggregate production function.
        params : dict
            Dictionary of model parameters.

        """
        self.output = output
        self.params = params

        # wrap the model system and jacobian (only need to do this once!)
        self._wrapped_sys = sp.lambdify(self._symbolic_args,
                                        self._symbolic_system,
                                        modules=[{'ImmutableMatrix': np.array}, "numpy"])
        self._wrapped_jac = sp.lambdify(self._symbolic_args,
                                        self._symbolic_jacobian,
                                        modules=[{'ImmutableMatrix': np.array}, "numpy"])

        super(Model, self).__init__(self._numeric_system, self._numeric_jacobian)

    @property
    def _symbolic_args(self):
        """Return list of symbolic arguments."""
        return [t, X] + sp.var(self.params.keys())

    @property
    def _symbolic_jacobian(self):
        """Symbolic Jacobian matrix of partial derivatives."""
        return self._symbolic_system.jacobian([X[0]])

    @property
    def _symbolic_system(self):
        """Symbolic system of ODE that define the model."""
        change_of_vars = {'k': X[0]}
        return sp.Matrix([self.k_dot]).subs(change_of_vars)

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

        Additional assumptions are that `f` satisfies :math:`f(0)=0`, is
        concave (i.e., :math:`f'(k) > 0, f''(k) < 0`), and satisfies the Inada
        conditions:

        .. math::

            \lim_{k \rigtharrow 0} = \infty \\
            \lim_{k \rightarrow \infty} = 0

        The Inada (1964) conditions are sufficient (but not necessary!) to
        ensure that the time path of capital per effective worker does not
        explode.

        :getter: Return the current intensive production function.
        :type: sp.Basic

        """
        return self._output.subs({'A': 1.0, 'K': k, 'L': 1.0})

    @property
    def k_dot(self):
        """
        Symbolic equation of motion for capital per effective worker.

        :getter: Return the current intensive production function.
        :type: sp.Basic

        """
        return s * self.intensive_output - (g + n + delta) * k

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
            Savings rate. Must satisfy `0 < s < 1`.
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

    @params.setter
    def params(self, value):
        """Set a new parameter dictionary."""
        self._params = value

    def _numeric_system(self, t, X):
        """
        Equation of motion for capital (per worker/effective worker) for a
        Solow growth model.

        Parameters
        ----------
        t : ndarray (float)
            Time.
        X : ndarray (float, shape=(1,))
            Endogenous variables of the Solow model. Ordering is `X = [k]`
            where `k` is capital (per worker/effective worker).

        Returns
        -------
        X_dot : ndarray (float, shape=(1,))
            Rate of change of capital (per worker/effective worker).

        """
        X_dot = self._wrapped_sys(t, X, **self.params).ravel()
        return X_dot

    def _numeric_jacobian(self, t, X):
        """
        Jacobian matrix of partial derivatives for the Solow model.

        Parameters
        ----------
        t : float
            Time.
        X : ndarray (float, shape=(1,))
            Endogenous variables of the Solow model. Ordering is `X = [k]`
            where `k` is capital (per worker/effective worker).

        Returns
        -------
        jac : array_like (float)
            Derivative of the equation of motion for capital (per worker/
            effective worker) with respect to `k`.

        """
        jac = self._wrapped_jac(t, X, **self.params)
        return jac

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
