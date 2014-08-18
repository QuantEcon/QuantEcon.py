r"""
======================
The Solow Growth Model
======================

This summary of the [solow1956] model of economic growth largely follows the
presentation found in [romer2011].

Assumptions
===========

The production function
----------------------------------------------

The [solow1956] model of economic growth focuses on the behavior of four
variables: output, `Y`, capital, `K`, labor, `L`, and knowledge (or technology
or the ``effectiveness of labor''), `A`. At each point in time the economy has
some amounts of capital, labor, and knowledge that can be combined to produce
output according to some production function, `F`.

.. math::

    Y(t) = F(K(t), A(t)L(t))

where `t` denotes time.

The evolution of the inputs to production
-----------------------------------------
The initial levels of capital, :math:`K_0`, labor, :math:`L_0`, and technology,
:math:`A_0`, are taken as given. Labor and technology are assumed to grow at
constant rates:

.. math::

    \dot{A}(t) = gA(t)
    \dot{L}(t) = nL(t)

where the rate of technological progrss, `g`, and the population growth rate,
`n`, are exogenous parameters.

Output is divided between consumption and investment. The fraction of output
devoted to investment, :math:`0 < s < 1`, is exogenous and constant. One unit
of output devoted to investment yields one unit of new capital. Capital is
assumed to decpreciate at a rate :math:`0\le \delta`. Thus aggregate capital
stock evolves according to

.. math::

    \dot{K}(t) = sY(t) - \delta K(t).

Although no restrictions are placed on the rates of technological progress and
population growth, the sum of `g`, `n`, and :math:`delta` is assumed to be
positive.

The dynamics of the model
=========================

Because the economy is growing over time (due to exogenous technological
progress and population growth) it is useful to focus on the behavior of
capital stock per unit of effective labor, :math:`k\equiv K/AL`. Applying
the chain rule to the equation of motion for capital stock yields (and a
bit of algebra!) yields an equation of motion for capital stock per unit of
effective labor.

.. math::

    \dot{k}(t) = s f(k) - (g + n + \delta)k(t)

References
==========
.. [romer2011] D. Romer. *Advanced Macroeconomics, 4th edition*, MacGraw Hill, 2011.
.. [solow1956] R. Solow. *A contribution to the theory of economic growth*, Quarterly Journal of Economics, 70(1):64-95, 1956.

@author : David R. Pugh
@date : 2014-08-18

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


class Model(object):

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
        # cached values
        self.__intensive_output = None

        self.output = output
        self.params = params

    @property
    def _intensive_output(self):
        """
        :getter: Return vectorized version of intensive aggregate production.
        :type: function

        """
        if self.__intensive_output is None:
            args = [k] + sp.var(self.params.keys())
            self.__intensive_output = sp.lambdify(args, self.intensive_output,
                                                  modules=[{'ImmutableMatrix': np.array}, "numpy"])
        return self.__intensive_output

    @property
    def intensive_output(self):
        r"""
        Symbolic expression for the intensive form of aggregate production.

        :getter: Return the current intensive production function.
        :type: sp.Basic

        Notes
        -----
        The assumption of constant returns to scale allows us to work the the
        intensive form of the aggregate production function, `F`. Defining
        :math:`c=1/AL` one can write

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

        The [inada1964]_ conditions are sufficient (but not necessary!) to
        ensure that the time path of capital per effective worker does not
        explode.

        .. [inada1964] K. Inda. *Some structural characteristics of Turnpike Theorems*,
        Review of Economic Studies, 31(1):43-58, 1964.

        """
        return self.output.subs({'A': 1.0, 'K': k, 'L': 1.0})

    @property
    def output(self):
        r"""
        Symbolic expression for the aggregate production function.

        :getter: Return the current aggregate production function.
        :setter: Set a new aggregate production function
        :type: sp.Basic

        Notes
        -----
        At each point in time the economy has some amounts of capital, `K`,
        labor, `L`, and knowledge (or technology), `A`, that can be combined to
        produce output, `Y`, according to some function, `F`.

        .. math::

            Y(t) = F(K(t), A(t)L(t))

        where `t` denotes time. Note that `A` and `L` are assumed to enter
        multiplicatively. Typically `A(t)L(t)` denotes "effective labor", and
        technology that enters in this fashion is known as labor-augmenting or
        "Harrod neutral."

        A key assumption of the model is that the function `F` exhibits
        constant returns to scale in capital and labor inputs. Specifically,

        .. math::

            F(cK(t), cA(t)L(t)) = cF(K(t), A(t)L(t)) = cY(t)

        for any :math:`c \ge 0`.

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

        # clear the cache
        self.__intensive_output = None

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

    def compute_actual_investment(self, k):
        """
        Return the amount of output (per worker/effective worker) invested in
        the production of new capital.

        Parameters
        ----------
        k : array_like (float)
            Capital (per worker/effective worker)

        Returns
        -------
        actual_inv : array_like (float)
            Investment (per worker/effective worker)

        """
        actual_inv = self.params['s'] * self.compute_intensive_output(k)
        return actual_inv

    def compute_effective_depreciation(self, k):
        """
        Return amount of capital (per worker/effective worker) that depreciates
        due to technological progress, population growth, and physical
        depreciation.

        Parameters
        ----------
        k : array_like (float)
            Capital (per worker/effective worker)

        Returns
        -------
        effective_dep : array_like (float)
            Amount of depreciated capital (per worker/effective worker)

        """
        effective_dep = self._effective_dep_rate * k
        return effective_dep

    def compute_intensive_output(self, k):
        """
        Return the amount of output (per worker/effective worker).

        Parameters
        ----------
        k : ndarray (float)
            Capital (per worker/effective worker)

        Returns
        -------
        y : ndarray (float)
            Output (per worker/effective worker)

        """
        y = self._intensive_output(k, **self.params)
        return y
