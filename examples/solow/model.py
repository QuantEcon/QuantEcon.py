r"""
======================
The Solow Growth Model
======================

The following summary of the [solow1956] model of economic growth
largely follows [romer2011].

Assumptions
===========

The production function
----------------------------------------------

The [solow1956] model of economic growth focuses on the behavior of four
variables: output, `Y`, capital, `K`, labor, `L`, and knowledge (or
technology or the ``effectiveness of labor''), `A`. At each point in
time the economy has some amounts of capital, labor, and knowledge that
can be combined to produce output according to some production function,
`F`.

.. math::

    Y(t) = F(K(t), A(t)L(t))

where `t` denotes time.

The evolution of the inputs to production
-----------------------------------------

The initial levels of capital, :math:`K_0`, labor, :math:`L_0`, and
technology, :math:`A_0`, are taken as given. Labor and technology are
assumed to grow at constant rates:

.. math::

    \dot{A}(t) = gA(t)
    \dot{L}(t) = nL(t)

where the rate of technological progrss, `g`, and the population growth
rate, `n`, are exogenous parameters.

Output is divided between consumption and investment. The fraction of
output devoted to investment, :math:`0 < s < 1`, is exogenous and
constant. One unit of output devoted to investment yields one unit of
new capital. Capital is assumed to decpreciate at a rate :math:`0\le
\delta`. Thus aggregate capital stock evolves according to

.. math::

    \dot{K}(t) = sY(t) - \delta K(t).

Although no restrictions are placed on the rates of technological
progress and population growth, the sum of `g`, `n`, and :math:`delta`
is assumed to be positive.

The dynamics of the model
=========================

Because the economy is growing over time (due to exogenous technological
progress and population growth) it is useful to focus on the behavior of
capital stock per unit of effective labor, :math:`k\equiv K/AL`.
Applying the chain rule to the equation of motion for capital stock
yields (after a bit of algebra!) an equation of motion for capital stock
per unit of effective labor.

.. math::

    \dot{k}(t) = s f(k) - (g + n + \delta)k(t)

References
==========
.. [romer2011] D. Romer. *Advanced Macroeconomics, 4th edition*, MacGraw Hill, 2011.
.. [solow1956] R. Solow. *A contribution to the theory of economic growth*, Quarterly Journal of Economics, 70(1):64-95, 1956.

@author : David R. Pugh
@date : 2014-11-27

"""
from __future__ import division
import collections
from textwrap import dedent

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
import sympy as sym

from ... import ivp
from . import impulse_response

# declare key variables for the model
t, X = sym.symbols('t'), sym.DeferredVector('X')
A, k, K, L, Y = sym.symbols('A, k, K, L, Y')

# declare required model parameters
g, n, s, delta = sym.symbols('g, n, s, delta')


class Model(object):

    __intensive_output = None

    __mpk = None

    __numeric_jacobian = None

    __numeric_solow_residual = None

    __numeric_system = None

    _modules = [{'ImmutableMatrix': np.array}, "numpy"]

    _required_params = ['g', 'n', 's', 'delta', 'A0', 'L0']

    def __init__(self, output, params):
        """
        Create an instance of the Solow growth model.

        Parameters
        ----------
        output : sym.Basic
            Symbolic expression defining the aggregate production
            function.
        params : dict
            Dictionary of model parameters.

        """
        self.irf = impulse_response.ImpulseResponse(self)
        self.output = output
        self.params = params

    def __repr__(self):
        """Machine readable summary of a Model instance."""
        return self.__str__()

    def __str__(self):
        """Human readable summary of a Model instance."""
        m = """
        Solow (1956) model of economic growth:
          - Output                                        : {Y}
          - A0 (initial level of technology)              : {A0:g}
          - L0 (initial amount of available labor)        : {L0:g}
          - g (growth rate of technology)                 : {g:g}
          - n (growth rate of the labor force)            : {n:g}
          - s (savings rate)                              : {s:g}
          - delta (depreciation rate of physical capital) : {delta:g}
        """
        formatted_str = dedent(m.format(Y=self.output,
                                        A0=self.params['A0'],
                                        L0=self.params['L0'],
                                        g=self.params['g'],
                                        n=self.params['n'],
                                        s=self.params['s'],
                                        delta=self.params['delta']))
        return formatted_str

    @property
    def _intensive_output(self):
        """
        :getter: Return vectorized symbolic intensive aggregate production.
        :type: function

        """
        if self.__intensive_output is None:
            args = [k] + sym.symbols(list(self.params.keys()))
            self.__intensive_output = sym.lambdify(args, self.intensive_output,
                                                   self._modules)
        return self.__intensive_output

    @property
    def _mpk(self):
        """
        :getter: Return vectorized symbolic marginal product capital.
        :type: function

        """
        if self.__mpk is None:
            args = [k] + sym.symbols(list(self.params.keys()))
            self.__mpk = sym.lambdify(args, self.marginal_product_capital,
                                      self._modules)
        return self.__mpk

    @property
    def _numeric_jacobian(self):
        """
        Vectorized, numpy-aware function defining the Jacobian matrix of
        partial derivatives.

        :getter: Return vectorized Jacobian matrix of partial derivatives.
        :type: function

        """
        if self.__numeric_jacobian is None:
            self.__numeric_jacobian = sym.lambdify(self._symbolic_args,
                                                   self._symbolic_jacobian,
                                                   self._modules)
        return self.__numeric_jacobian

    @property
    def _numeric_solow_residual(self):
        """
        Vectorized, numpy-aware function defining the Solow residual.

        :getter: Return vectorized symbolic Solow residual.
        :type: function

        """
        if self.__numeric_solow_residual is None:
            tmp_args = [Y, K, L] + sym.symbols(list(self.params.keys()))
            self.__numeric_solow_residual = sym.lambdify(tmp_args,
                                                         self.solow_residual,
                                                         self._modules)
        return self.__numeric_solow_residual

    @property
    def _numeric_system(self):
        """
        Vectorized, numpy-aware function defining the system of ODEs.

        :getter: Return vectorized symbolic system of ODEs.
        :type: function

        """
        if self.__numeric_system is None:
            self.__numeric_system = sym.lambdify(self._symbolic_args,
                                                 self._symbolic_system,
                                                 self._modules)
        return self.__numeric_system

    @property
    def _symbolic_args(self):
        """
        List of symbolic arguments used in constructing vectorized
        versions of _symbolic_system and _symbolic_jacobian.

        :getter: Return list of symbolic arguments.
        :type: list

        """
        args = [t, X] + sym.symbols(list(self.params.keys()))
        return args

    @property
    def _symbolic_jacobian(self):
        """
        Symbolic Jacobian matrix for the system of ODEs.

        :getter: Return the symbolic Jacobian matrix.
        :type: sym.MutableDenseMatrix

        """
        N = self._symbolic_system.shape[0]
        return self._symbolic_system.jacobian([X[i] for i in range(N)])

    @property
    def _symbolic_system(self):
        """
        Symbolic matrix defining the system of ODEs.

        :getter: Return the matrix defining the system of ODEs.
        :type: sym.MutableDenseMatrix

        """
        change_of_vars = {k: X[0]}
        return sym.Matrix([self.k_dot]).subs(change_of_vars)

    @property
    def effective_depreciation_rate(self):
        """
        Effective depreciation rate for capital stock (per unit
        effective labor).

        :getter: Return the current effective depreciation rate.
        :type: float

        Notes
        -----
        The effective depreciation rate of physical capital takes into
        account both technological progress and population growth, as
        well as physical depreciation.

        """
        return sum(self.params[key] for key in ['g', 'n', 'delta'])

    @property
    def intensive_output(self):
        r"""
        Symbolic expression for the intensive form of aggregate
        production.

        :getter: Return the current intensive production function.
        :type: sym.Basic

        Notes
        -----
        The assumption of constant returns to scale allows us to work
        the the intensive form of the aggregate production function,
        `F`. Defining :math:`c=1/AL` one can write

        ..math::

            F\bigg(\frac{K}{AL}, 1\bigg) = \frac{1}{AL}F(A, K, L)

        Defining :math:`k=K/AL` and :math:`y=Y/AL` to be capital per
        unit effective labor and output per unit effective labor,
        respectively, the intensive form of the production function can
        be written as

        .. math::

            y = f(k).

        Additional assumptions are that `f` satisfies :math:`f(0)=0`, is
        concave (i.e., :math:`f'(k) > 0, f''(k) < 0`), and satisfies the
        Inada conditions:

        .. math::
            :type: eqnarray

            \lim_{k \rightarrow 0} &=& \infty \\
            \lim_{k \rightarrow \infty} &=& 0

        The [inada1964]_ conditions are sufficient (but not necessary!)
        to ensure that the time path of capital per effective worker
        does not explode.

        .. [inada1964] K. Inda. *Some structural characteristics of Turnpike Theorems*, Review of Economic Studies, 31(1):43-58, 1964.

        """
        return self.output.subs({'A': 1.0, 'K': k, 'L': 1.0})

    @property
    def ivp(self):
        r"""
        Initial value problem

        :getter: Return an instance of the ivp.IVP class representing
        the Solow model.
        :type: ivp.IVP

        Notes
        -----
        The Solow model with can be formulated as an initial value
        problem (IVP) as follows.

        .. math::

            \dot{k}(t) = sf(k(t)) - (g + n + \delta)k(t),\ t\ge t_0,\ k(t_0) = k_0

        The solution to this IVP is a function :math:`k(t)` describing
        the time path of capital stock (per unit effective labor).

        """
        tmp_ivp = ivp.IVP(self._numeric_system, self._numeric_jacobian)
        tmp_ivp.f_params = tuple(self.params.values())
        tmp_ivp.jac_params = tuple(self.params.values())
        return tmp_ivp

    @property
    def k_dot(self):
        r"""
        Symbolic expression for the equation of motion for capital (per
        unit effective labor).

        :getter: Return the current equation of motion for capital (per
        unit effective labor).
        :type: sym.Basic

        Notes
        -----
        Because the economy is growing over time due to technological
        progress, `g`, and population growth, `n`, it makes sense to
        focus on the capital stock per unit effective labor, `k`, rather
        than aggregate physical capital, `K`. Since, by definition,
        :math:`k=K/AL`, we can apply the chain rule to the time derative
        of `k`.

        .. math::
            :type: eqnarray

            \dot{k}(t) =& \frac{\dot{K}(t)}{A(t)L(t)} - \frac{K(t)}{[A(t)L(t)]^2}\bigg[\dot{A}(t)L(t) + \dot{L}(t)A(t)\bigg] \\
            =& \frac{\dot{K}(t)}{A(t)L(t)} - \bigg(\frac{\dot{A}(t)}{A(t)} + \frac{\dot{L}(t)}{L(t)}\bigg)\frac{K(t)}{A(t)L(t)}

        By definition, math:`k=K/AL`, and by assumption
        :math:`\dot{A}/A` and :math:`\dot{L}/L` are `g` and `n`
        respectively. Aggregate capital stock evolves according to

        .. math::

            \dot{K}(t) = sF(K(t), A(t)L(t)) - \delta K(t).

        Substituting these facts into the above equation yields the
        equation of motion for capital stock (per unit effective labor).

        .. math::
            :type: eqnarray

            \dot{k}(t) =& \frac{sF(K(t), A(t)L(t)) - \delta K(t)}{A(t)L(t)} - (g + n)k(t) \\
            =& \frac{sY(t)}{A(t)L(t)} - (g + n + \delta)k(t) \\
            =& sf(k(t)) - (g + n + \delta)k(t)

        """
        return s * self.intensive_output - (g + n + delta) * k

    @property
    def marginal_product_capital(self):
        r"""
        Symbolic expression for the marginal product of capital (per
        unit effective labor).

        :getter: Return the current marginal product of capital (per
        unit effective labor).
        :type: sym.Basic

        Notes
        -----
        The marginal product of capital is defined as follows:

        .. math::

            \frac{\partial F(K, AL)}{\partial K} \equiv f'(k)

        where :math:`k=K/AL` is capital stock (per unit effective labor)

        """
        return sym.diff(self.intensive_output, k)

    @property
    def output(self):
        r"""
        Symbolic expression for the aggregate production function.

        :getter: Return the current aggregate production function.
        :setter: Set a new aggregate production function
        :type: sym.Basic

        Notes
        -----
        At each point in time the economy has some amounts of capital,
        `K`, labor, `L`, and knowledge (or technology), `A`, that can be
        combined to produce output, `Y`, according to some function,
        `F`.

        .. math::

            Y(t) = F(K(t), A(t)L(t))

        where `t` denotes time. Note that `A` and `L` are assumed to
        enter multiplicatively. Typically `A(t)L(t)` denotes "effective
        labor", and technology that enters in this fashion is known as
        labor-augmenting or "Harrod neutral."

        A key assumption of the model is that the function `F` exhibits
        constant returns to scale in capital and labor inputs.
        Specifically,

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

        A0: float
            Initial level of technology. Must satisfy :math:`A_0 > 0 `.
        L0: float
            Initial amount of available labor. Must satisfy
            :math:`L_0 > 0 `.
        g : float
            Growth rate of technology.
        n : float
            Growth rate of the labor force.
        s : float
            Savings rate. Must satisfy `0 < s < 1`.
        delta : float
            Depreciation rate of physical capital. Must satisfy
            :math:`0 < \delta`.

        Although no restrictions are placed on the rates of
        technological progress and population growth, the sum of `g`,
        `n`, and :math:`delta` is assumed to be positive. The user mus
        also specify any additional model parameters specific to the
        chosen aggregate production function.

        """
        return self._params

    @property
    def solow_residual(self):
        """
        Symbolic expression for the Solow residual which is used as a
        measure of technology.

        :getter: Return the symbolic expression.
        :type: sym.Basic

        """
        return sym.solve(Y - self.output, A)[0]

    @property
    def speed_of_convergence(self):
        r"""
        The speed of convergence for the Solow model.

        :getter: Return the current speed of convergence.
        :type: float

        Notes
        -----
        The following is a derivation for the speed of convergence
        :math:`\lambda`:

        .. :math::
            :type: eqnarray

            \lambda \equiv -\frac{\partial \dot{k}(k(t))}{\partial k(t)}\bigg|_{k(t)=k^*} =& -[sf'(k^*) - (g + n+ \delta)] \\
            =& (g + n+ \delta) - sf'(k^*) \\
            =& (g + n + \delta) - (g + n + \delta)\frac{k^*f'(k^*)}{f(k^*)} \\
            =& (1 - \alpha_K(k^*))(g + n + \delta)

        where the elasticity of output with respect to capital,
        $\alpha_K(k)$, is defined as

        .. :math::

            \alpha_K(k) = \frac{k'(k)}{f(k)}.

        """
        alpha_K = self.evaluate_output_elasticity(self.steady_state)
        return (1 - alpha_K) * self.effective_depreciation_rate

    @property
    def steady_state(self):
        r"""
        Steady state value of capital stock (per unit effective labor).

        :getter: Return the current steady state value.
        :type: float

        Notes
        -----
        The steady state value of capital stock (per unit effective
        labor), `k`, is defined as the value of `k` that solves

        .. math::

            0 = sf(k) - (g + n + \delta)k

        where `s` is the savings rate, `f(k)` is intensive output, and
        :math:`g + n + \delta` is the effective depreciation rate.

        """
        lower, upper = 1e-12, 1e12
        return self.find_steady_state(lower, upper)

    @output.setter
    def output(self, value):
        """Set a new production function."""
        self._output = self._validate_output(value)
        self._clear_cache()

    @params.setter
    def params(self, value):
        """Set a new parameter dictionary."""
        valid_params = self._validate_params(value)
        self._params = self._order_params(valid_params)

    def _clear_cache(self):
        """Clear cached values."""
        self.__intensive_output = None
        self.__mpk = None
        self.__numeric_jacobian = None
        self.__numeric_solow_residual = None
        self.__numeric_system = None

    @staticmethod
    def _order_params(params):
        """Cast a dictionary to an order dictionary."""
        return collections.OrderedDict(sorted(params.items()))

    def _validate_output(self, output):
        """Validate the production function."""
        if not isinstance(output, sym.Basic):
            mesg = ("Output must be an instance of {}.".format(sym.Basic))
            raise AttributeError(mesg)
        elif not ({A, K, L} < output.atoms()):
            mesg = ("Output must be an expression of technology, 'A', " +
                    "capital, 'K', and labor, 'L'.")
            raise AttributeError(mesg)
        else:
            return output

    def _validate_params(self, params):
        """Validate the model parameters."""
        if not isinstance(params, dict):
            mesg = "SolowModel.params must be a dict, not a {}."
            raise AttributeError(mesg.format(params.__class__))
        elif not set(self._required_params) <= set(params.keys()):
            mesg = "One of the required params in {} has not been specified."
            raise AttributeError(mesg.format(self._required_params))
        elif params['s'] <= 0.0 or params['s'] >= 1.0:
            raise AttributeError('Savings rate must be in (0, 1).')
        elif params['delta'] <= 0.0 or params['delta'] >= 1.0:
            raise AttributeError('Depreciation rate must be in (0, 1).')
        elif params['g'] + params['n'] + params['delta'] <= 0.0:
            raise AttributeError("Sum of g, n, and delta must be positive.")
        elif params['A0'] <= 0.0:
            mesg = "Initial value for technology must be strictly positive."
            raise AttributeError(mesg)
        elif params['L0'] <= 0.0:
            mesg = "Initial value for labor supply must be strictly positive."
            raise AttributeError(mesg)
        else:
            return params

    def evaluate_actual_investment(self, k):
        """
        Return the amount of output (per unit of effective labor)
        invested in the production of new capital.

        Parameters
        ----------
        k : array_like (float)
            Capital stock (per unit of effective labor)

        Returns
        -------
        actual_inv : array_like (float)
            Investment (per unit of effective labor)

        """
        actual_inv = self.params['s'] * self.evaluate_intensive_output(k)
        return actual_inv

    def evaluate_consumption(self, k):
        """
        Return the amount of consumption (per unit of effective labor).

        Parameters
        ----------
        k : ndarray (float)
            Capital stock (per unit of effective labor)

        Returns
        -------
        c : ndarray (float)
            Consumption (per unit of effective labor)

        """
        c = (self.evaluate_intensive_output(k) -
             self.evaluate_actual_investment(k))
        return c

    def evaluate_effective_depreciation(self, k):
        """
        Return amount of Capital stock (per unit of effective labor)
        that depreciaties due to technological progress, population
        growth, and physical depreciation.

        Parameters
        ----------
        k : array_like (float)
            Capital stock (per unit of effective labor)

        Returns
        -------
        effective_depreciation : array_like (float)
            Amount of depreciated Capital stock (per unit of effective
            labor)

        """
        effective_depreciation = self.effective_depreciation_rate * k
        return effective_depreciation

    def evaluate_intensive_output(self, k):
        """
        Return the amount of output (per unit of effective labor).

        Parameters
        ----------
        k : ndarray (float)
            Capital stock (per unit of effective labor)

        Returns
        -------
        y : ndarray (float)
            Output (per unit of effective labor)

        """
        y = self._intensive_output(k, *self.params.values())
        return y

    def evaluate_k_dot(self, k):
        """
        Return time derivative of capital stock (per unit of effective
        labor).

        Parameters
        ----------
        k : ndarray (float)
            Capital stock (per unit of effective labor)

        Returns
        -------
        k_dot : ndarray (float)
            Time derivative of capital stock (per unit of effective
            labor).

        """
        k_dot = (self.evaluate_actual_investment(k) -
                 self.evaluate_effective_depreciation(k))
        return k_dot

    def evaluate_mpk(self, k):
        """
        Return marginal product of capital stock (per unit of effective
        labor).

        Parameters
        ----------
        k : ndarray (float)
            Capital stock (per unit of effective labor)

        Returns
        -------
        mpk : ndarray (float)
            Marginal product of capital stock (per unit of effective
            labor).

        """
        mpk = self._mpk(k, *self.params.values())
        return mpk

    def evaluate_output_elasticity(self, k):
        """
        Return elasticity of output with respect to capital stock (per
        unit effective labor).

        Parameters
        ----------
        k : array_like (float)
            Capital stock (per unit of effective labor)

        Returns
        -------
        alpha_k : array_like (float)
            Elasticity of output with respect to capital stock (per unit
            effective labor).

        Notes
        -----
        Under the additional assumption that markets are perfectly
        competitive, the elasticity of output with respect to capital
        stock is equivalent to capital's share of income. Since, under
        perfect competition, firms earn zero profits it must be true
        capital's share and labor's share must sum to one.

        """
        alpha_k = (k*self.evaluate_mpk(k)) / self.evaluate_intensive_output(k)
        return alpha_k

    def evaluate_solow_residual(self, Y, K, L):
        """
        Return Solow residual.

        Parameters
        ----------
        k : array_like (float)
            Capital stock (per unit of effective labor)

        Returns
        -------
        residual : array_like (float)
            Solow residual

        """
        residual = self._numeric_solow_residual(Y, K, L, *self.params.values())
        assert residual.all() > 0, "Solow residual show always be positive!"
        return residual

    def find_steady_state(self, a, b, method='brentq', **kwargs):
        """
        Compute the equilibrium value of capital stock (per unit
        effective labor).

        Parameters
        ----------
        a : float
            One end of the bracketing interval [a,b].
        b : float
            The other end of the bracketing interval [a,b]
        method : str (default=`brentq`)
            Method to use when computing the steady state. Supported
            methods are `bisect`, `brenth`, `brentq`, `ridder`. See
            `scipy.optimize` for more details (including references).
        kwargs : optional
            Additional keyword arguments. Keyword arguments are method
            specific see `scipy.optimize` for details.

        Returns
        -------
        x0 : float
            Zero of `f` between `a` and `b`.
        r : RootResults (present if ``full_output = True``)
            Object containing information about the convergence. In
            particular, ``r.converged`` is True if the routine
            converged.

        """
        if method == 'bisect':
            result = optimize.bisect(self.evaluate_k_dot, a, b, **kwargs)
        elif method == 'brenth':
            result = optimize.brenth(self.evaluate_k_dot, a, b, **kwargs)
        elif method == 'brentq':
            result = optimize.brentq(self.evaluate_k_dot, a, b, **kwargs)
        elif method == 'ridder':
            result = optimize.ridder(self.evaluate_k_dot, a, b, **kwargs)
        else:
            mesg = ("Method must be one of : 'bisect', 'brenth', 'brentq', " +
                    "or 'ridder'.")
            raise ValueError(mesg)

        return result

    def linearized_solution(self, t, k0):
        """
        Compute the linearized solution for the Solow model.

        Parameters
        ----------
        t : ndarray (shape=(T,))
            Array of points at which the solution is desired.
        k0 : (float)
            Initial condition for capital stock (per unit of effective
            labor)

        Returns
        -------
        linearized_traj : ndarray (shape=t.size, 2)
            Array representing the linearized solution trajectory.

        """
        kt = (self.steady_state + np.exp(-self.speed_of_convergence * t) *
              (k0 - self.steady_state))
        linearized_traj = np.hstack((t[:, np.newaxis], kt[:, np.newaxis]))

        return linearized_traj

    def plot_factor_shares(self, ax, Nk=1e3, **new_params):
        """
        Plot income/output shares of capital and labor inputs to
        production.

        Parameters
        ----------
        ax : `matplotlib.axes.AxesSubplot`
            An instance of `matplotlib.axes.AxesSubplot`.
        Nk : float (default=1e3)
            Number of capital stock (per unit of effective labor) grid
            points.
        new_params : dict (optional)
            Optional dictionary of parameter values to change.

        Returns
        -------
        A list containing...

        capitals_share_line : maplotlib.lines.Line2D
            A Line2D object representing the time path for capital's
            share of income.
        labors_share_line : maplotlib.lines.Line2D
            A Line2D object representing the time path for labor's
            share of income.

        """
        # create tmp_params dict to force check for valid params
        tmp_params = self.params.copy()
        tmp_params.update(new_params)
        self.params = tmp_params  # forces check for valid params!

        # create the plot
        k_grid = np.linspace(0, 2 * self.steady_state, Nk)
        capitals_share = self.evaluate_output_elasticity(k_grid)
        labors_share = 1 - capitals_share

        capitals_share_line, = ax.plot(k_grid, capitals_share, 'r-',
                                       label=r'$\alpha_K(k(t))$')
        labors_share_line, = ax.plot(k_grid, labors_share, 'b-',
                                     label=r'$1 - \alpha_K(k(t))$')
        ax.set_xlabel('Capital (per unit effective labor), $k(t)$',
                      family='serif', fontsize=15)
        ax.set_title('Factor shares', family='serif', fontsize=20)
        ax.grid(True)
        ax.legend(loc=0, frameon=False, prop={'family': 'serif'},
                  bbox_to_anchor=(1.0, 1.0))

        return [capitals_share_line, labors_share_line]

    def plot_intensive_output(self, ax, Nk=1e3, **new_params):
        """
        Plot intensive form of the aggregate production function.

        Parameters
        ----------
        ax : `matplotlib.axes.AxesSubplot`
            An instance of `matplotlib.axes.AxesSubplot`.
        Nk : float (default=1e3)
            Number of capital stock (per unit of effective labor) grid
            points.
        new_params : dict (optional)
            Optional dictionary of parameter values to change.

        Returns
        -------
        A list containing...

        intensive_output : maplotlib.lines.Line2D
            A Line2D object representing intensive output as a function
            of capital stock (per unit effective labor).

        """
        # create tmp_params dict to force check for valid params
        tmp_params = self.params.copy()
        tmp_params.update(new_params)
        self.params = tmp_params  # forces check for valid params!

        # create the plot
        k_grid = np.linspace(0, 2 * self.steady_state, Nk)
        y_grid = self.evaluate_intensive_output(k_grid)
        intensive_output_line, = ax.plot(k_grid, y_grid, 'r-')
        ax.set_xlabel('Capital (per unit effective labor), $k(t)$',
                      family='serif', fontsize=15)
        ax.set_ylabel('$f(k(t))$', family='serif', fontsize=20,
                      rotation='horizontal')
        ax.yaxis.set_label_coords(-0.1, 0.5)
        ax.set_title('Output (per unit effective labor)',
                     family='serif', fontsize=20)
        ax.grid(True)

        return [intensive_output_line]

    def plot_intensive_investment(self, ax, Nk=1e3, **new_params):
        """
        Plot actual investment (per unit effective labor) and effective
        depreciation. The steady state value of capital stock (per unit
        effective labor) balance acual investment and effective
        depreciation.

        Parameters
        ----------
        ax : `matplotlib.axes.AxesSubplot`
            An instance of `matplotlib.axes.AxesSubplot`.
        Nk : float (default=1e3)
            Number of capital stock (per unit of effective labor) grid
            points.
        new_params : dict (optional)
            Optional dictionary of parameter values to change.

        Returns
        -------
        A list containing...

        actual_investment_line : maplotlib.lines.Line2D
            A Line2D object representing the level of actual investment
            as a function of capital stock (per unit effective labor).
        breakeven_investment_line : maplotlib.lines.Line2D
            A Line2D object representing the "break-even" level of
            investment as a function of capital stock (per unit
            effective labor).
        ss_line : maplotlib.lines.Line2D
            A Line2D object representing the steady state level of
            investment.

        """
        # create tmp_params dict to force check for valid params
        tmp_params = self.params.copy()
        tmp_params.update(new_params)
        self.params = tmp_params  # forces check for valid params!

        # create the plot
        k_grid = np.linspace(0, 2 * self.steady_state, Nk)
        actual_investment_grid = self.evaluate_actual_investment(k_grid)
        breakeven_investment_grid = self.evaluate_effective_depreciation(k_grid)
        ss_investment = self.evaluate_actual_investment(self.steady_state)

        actual_investment_line, = ax.plot(k_grid, actual_investment_grid, 'g-',
                                          label='$sf(k(t))$')
        breakeven_investment_line, = ax.plot(k_grid, breakeven_investment_grid,
                                             'b-', label='$(g + n + \delta)k(t)$')
        ss_line, = ax.plot(self.steady_state, ss_investment, 'ko',
                           label='$k^*={0:.4f}$'.format(self.steady_state))
        ax.set_xlabel('Capital (per unit effective labor), $k(t)$',
                      family='serif', fontsize=15)
        ax.set_ylabel('Investment (per unit effective labor)', family='serif',
                      fontsize=15)
        ax.set_title('Output (per unit effective labor)',
                     family='serif', fontsize=20)
        ax.grid(True)
        ax.legend(loc=0, frameon=False, prop={'family': 'serif'},
                  bbox_to_anchor=(1.0, 1.0))

        return [actual_investment_line, breakeven_investment_line, ss_line]

    def plot_phase_diagram(self, ax, Nk=1e3, **new_params):
        """
        Plot the model's phase diagram.

        Parameters
        ----------
        ax : `matplotlib.axes.AxesSubplot`
            An instance of `matplotlib.axes.AxesSubplot`.
        Nk : float (default=1e3)
            Number of capital stock (per unit of effective labor) grid
            points.
        new_params : dict (optional)
            Optional dictionary of parameter values to change.

        Returns
        -------
        A list containing...

        k_dot_line : maplotlib.lines.Line2D
            A Line2D object representing the rate of change of capital
            stock (per unit effective labor) as a function of its level.
        origin_line : maplotlib.lines.Line2D
            A Line2D object representing the origin (i.e., locus of
            points where k_dot is zero).
        ss_line : maplotlib.lines.Line2D
            A Line2D object representing the steady state level of
            capital stock (per unit effective labor).

        """
        # create tmp_params dict to force check for valid params
        tmp_params = self.params.copy()
        tmp_params.update(new_params)
        self.params = tmp_params  # forces check for valid params!

        # create the plot
        k_grid = np.linspace(0, 2 * self.steady_state, Nk)
        k_dot_line, = ax.plot(k_grid, self.evaluate_k_dot(k_grid),
                              color='orange')
        origin_line = ax.axhline(0, color='k')
        ss_line, = ax.plot(self.steady_state, 0.0, 'ko',
                           label='$k^*={0:.4f}$'.format(self.steady_state))
        ax.set_xlabel('Capital (per unit effective labor), $k(t)$',
                      family='serif', fontsize=15)
        ax.set_ylabel('$\dot{k}(t)$', family='serif', fontsize=25,
                      rotation='horizontal')
        ax.yaxis.set_label_coords(-0.1, 0.5)
        ax.set_title('Phase diagram', family='serif', fontsize=20)
        ax.grid(True)

        return [k_dot_line, origin_line, ss_line]

    def plot_solow_diagram(self, ax, Nk=1e3, **new_params):
        """
        Plot the classic Solow diagram.

        Parameters
        ----------
        ax : `matplotlib.axes.AxesSubplot`
            An instance of `matplotlib.axes.AxesSubplot`.
        Nk : float (default=1e3)
            Number of capital stock (per unit of effective labor) grid
            points.
        new_params : dict (optional)
            Optional dictionary of parameter values to change.

        Returns
        -------
        A list containing...

        actual_investment_line : maplotlib.lines.Line2D
            A Line2D object representing the level of actual investment
            as a function of capital stock (per unit effective labor).
        breakeven_investment_line : maplotlib.lines.Line2D
            A Line2D object representing the "break-even" level of
            investment as a function of capital stock (per unit
            effective labor).
        ss_line : maplotlib.lines.Line2D
            A Line2D object representing the steady state level of
            investment.

        """
        # create tmp_params dict to force check for valid params
        tmp_params = self.params.copy()
        tmp_params.update(new_params)
        self.params = tmp_params  # forces check for valid params!

        # create the plot
        k_grid = np.linspace(0, 2 * self.steady_state, Nk)
        intensive_output_grid = self.evaluate_intensive_output(k_grid)
        actual_investment_grid = self.evaluate_actual_investment(k_grid)
        breakeven_investment_grid = self.evaluate_effective_depreciation(k_grid)
        ss_investment = self.evaluate_actual_investment(self.steady_state)

        intensive_output_line, = ax.plot(k_grid, intensive_output_grid, 'r-',
                                         label='$f(k(t)$')
        actual_investment_line, = ax.plot(k_grid, actual_investment_grid, 'g-',
                                          label='$sf(k(t))$')
        breakeven_investment_line, = ax.plot(k_grid, breakeven_investment_grid,
                                             'b-', label='$(g + n + \delta)k(t)$')
        ss_line, = ax.plot(self.steady_state, ss_investment, 'ko',
                           label='$k^*={0:.4f}$'.format(self.steady_state))
        ax.set_xlabel('Capital (per unit effective labor), $k(t)$',
                      family='serif', fontsize=15)
        ax.set_title('Solow diagram',
                     family='serif', fontsize=20)
        ax.grid(True)
        ax.legend(loc=0, frameon=False, prop={'family': 'serif'},
                  bbox_to_anchor=(1, 1))

        lines = [intensive_output_line, actual_investment_line,
                 breakeven_investment_line, ss_line]

        return lines
