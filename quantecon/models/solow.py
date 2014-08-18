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
from IPython.html.widgets import *

import matplotlib.pyplot as plt
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
    def _effective_depreciation_rate(self):
        """Effective depreciation rate for physical capital."""
        return sum(self.params[key] for key in ['g', 'n', 'delta'])

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

        .. [inada1964] K. Inda. *Some structural characteristics of Turnpike Theorems*, Review of Economic Studies, 31(1):43-58, 1964.

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

        :getter: Return the current dictionary of model parameters.
        :setter: Set a new dictionary of model parameters.
        :type: dict

        Notes
        -----
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

        Although no restrictions are placed on the rates of technological
        progress and population growth, the sum of `g`, `n`, and :math:`delta`
        is assumed to be positive. The user mus also specify any additional
        model parameters specific to the chosen aggregate production function.

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
        self._params = self._validate_params(value)

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
        effective_depreciation : array_like (float)
            Amount of depreciated capital (per worker/effective worker)

        """
        effective_depreciation = self._effective_depreciation_rate * k
        return effective_depreciation

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


def plot_intensive_output(cls, k_upper=10, **new_params):
    """
    Plot intensive form of the aggregate production function.

    Parameters
    ----------
    cls : object
        An instance of :class:`quantecon.models.solow.Model`.
    k_upper : float
        Upper bound on capital stock (per unit of effective labor)
    new_params : dict (optional)
        Optional dictionary of parameter values to change.

    Returns
    -------
    A list containing:

    fig : object
        An instance of :class:`matplotlib.figure.Figure`.
    ax : object
        An instance of :class:`matplotlib.axes.AxesSubplot`.

    """

    # update model parameters
    cls.params.update(new_params)

    # create the plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8), squeeze=True)
    k_grid = np.linspace(0, k_upper, 1e3)
    ax.plot(k_grid, cls.compute_intensive_output(k_grid), 'r-')
    ax.set_xlabel('Capital (per unit effective labor), $k$', family='serif',
                  fontsize=15)
    ax.set_ylabel('$f(k)$', family='serif', fontsize=25,
                  rotation='horizontal')
    ax.set_title('Output (per unit effective labor)',
                 family='serif', fontsize=20)
    ax.grid(True)

    return [fig, ax]


def plot_intensive_invesment(cls, k_upper=10, **new_params):
    """
    Plot actual investment (per unit effective labor) and effective
    depreciation. The steady state value of capital stock (per unit effective
    labor) balance acual investment and effective depreciation.

    Parameters
    ----------
    cls : object
        An instance of :class:`quantecon.models.solow.Model`.
    k_upper : float
        Upper bound on capital stock (per unit of effective labor)
    new_params : dict (optional)
        Optional dictionary of parameter values to change.

    Returns
    -------
    A list containing:

    fig : object
        An instance of :class:`matplotlib.figure.Figure`.
    ax : object
        An instance of :class:`matplotlib.axes.AxesSubplot`.

    """
    # update the model parameters
    cls.params.update(new_params)

    # create the plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8), squeeze=True)
    k_grid = np.linspace(0, k_upper, 1e3)
    ax.plot(k_grid, cls.compute_actual_investment(k_grid), 'g-',
            label='$sf(k)$')
    ax.plot(k_grid, cls.compute_effective_depreciation(k_grid), 'b-',
            label='$(g + n + \delta)k$')
    ax.set_xlabel('Capital (per unit effective labor), $k$', family='serif',
                  fontsize=15)
    ax.set_ylabel('Investment (per unit effective labor)', family='serif',
                  fontsize=15)
    ax.set_title('Output (per unit effective labor)',
                 family='serif', fontsize=20)
    ax.grid(True)
    ax.legend(loc=0, frameon=False, prop={'family': 'serif'})

    return [fig, ax]


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
