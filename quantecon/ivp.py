from __future__ import division

import numpy as np
from scipy import integrate, interpolate


class IVP(object):
    """
    Base class for solving initial value problems (IVPs) of the form:

    :math:`y'(t) = f(t,y)`

    using finite difference methods. The class uses various integrators from
    the ``scipy.ode`` module to perform the integration and parametric B-spline
    interpolation from ``scipy.interpolate`` to approximate the value of the
    solution between grid points.

    """

    def __init__(self, f, jac=None, args=None):
        """
        Creates an instance of the IVP class.

        Attributes
        ----------
        f : callable ``f(t, y, *args)``
            Right hand side of the system of equations defining the ODE. The
            independent variable, `t`, is a ``scalar``; `y` is an ``ndarray``
            of dependent variables with ``y.shape == (n,)``. The function `f`
            should return a ``scalar``, ``ndarray`` or ``list`` (but not a
            ``tuple``).
        jac : callable ``jac(t, y, *args)``, optional(default=None)
            Jacobian of the right hand side of the system of equations defining
            the ODE.
            :math:`\mathcal{J}_{i,j} = \frac{\partial f_i}}{\partial y_j}`
        args : tuple, optional(default=None)
            Additional arguments that should be passed to both `f` and `jac`.

        """
        self.f = f
        self.jac = jac
        self.args = args
        self.ode = integrate.ode(f, jac)

    def integrate(self, t0, y0, h=1.0, T=None, g=None, tol=None,
                  integrator='dopri5', step=False, relax=False, **kwargs):
        """
        Integrates the ODE given some initial condition.

        Parameters
        ----------
            t0 : float
                Initial condition for the independent variable.
            y0 : array_like (float, shape=(n,))
                Initial condition for the dependent variables.
            h : float, optional(default=1.0)
                Step-size for computing the solution. Can be positive or
                negative depending on the desired direction of integration.
            T : int, optional(default=None)
                Terminal value for the independent variable. One of either `T`
                or `g` must be specified.
            g : callable ``g(t, vec, args)``, optional(default=None)
                Provides a stopping condition for the integration. If specified
                user must also specify a stopping tolerance, `tol`.
            tol : float, optional (default=None)
                Stopping tolerance for the integration. Only required if `g` is
                also specifed.
            integrator : str, optional(default='dopri5')
                Must be one of 'vode', 'lsoda', 'dopri5', or 'dop853'
            step : bool, optional(default=False)
                Allows access to internal steps for those solvers that use
                adaptive step size routines. Currently only 'vode', 'zvode',
                and 'lsoda' support `step=True`.
            relax : bool, optional(default=False)
                Currently only 'vode', 'zvode', and 'lsoda' support
                `relax=True`.
            **kwargs : dict, optional(default=None)
                Dictionary of integrator specific keyword arguments. See the
                Notes section below for a detailed discussion of the valid
                keyword arguments for each of the supported integrators.

        Notes
        -----
        Descriptions of the available integrators are listed below.

        "vode"

            Real-valued Variable-coefficient Ordinary Differential Equation
            solver, with fixed-leading-coefficient implementation. It provides
            implicit Adams method (for non-stiff problems) and a method based
            on backward differentiation formulas (BDF) (for stiff problems).

            Source: http://www.netlib.org/ode/vode.f

            .. warning::

               This integrator is not re-entrant. You cannot have two `ode`
               instances using the "vode" integrator at the same time.

            This integrator accepts the following keyword arguments:

            - atol : float or sequence
              absolute tolerance for solution
            - rtol : float or sequence
              relative tolerance for solution
            - lband : None or int
            - rband : None or int
              Jacobian band width, jac[i,j] != 0 for i-lband <= j <= i+rband.
              Setting these requires your jac routine to return the jacobian
              in packed format, jac_packed[i-j+lband, j] = jac[i,j].
            - method: 'adams' or 'bdf'
              Which solver to use, Adams (non-stiff) or BDF (stiff)
            - with_jacobian : bool
              Whether to use the jacobian
            - nsteps : int
              Maximum number of (internally defined) steps allowed during one
              call to the solver.
            - first_step : float
            - min_step : float
            - max_step : float
              Limits for the step sizes used by the integrator.
            - order : int
              Maximum order used by the integrator,
              order <= 12 for Adams, <= 5 for BDF.

        "zvode"

            Complex-valued Variable-coefficient Ordinary Differential Equation
            solver, with fixed-leading-coefficient implementation.  It provides
            implicit Adams method (for non-stiff problems) and a method based
            on backward differentiation formulas (BDF) (for stiff problems).

            Source: http://www.netlib.org/ode/zvode.f

            .. warning::

               This integrator is not re-entrant. You cannot have two `ode`
               instances using the "zvode" integrator at the same time.

            This integrator accepts the same keyword arguments as "vode".

            .. note::

                When using ZVODE for a stiff system, it should only be used for
                the case in which the function f is analytic, that is, when
                each f(i) is an analytic function of each y(j).  Analyticity
                means that the partial derivative df(i)/dy(j) is a unique
                complex number, and this fact is critical in the way ZVODE
                solves the dense or banded linear systems that arise in the
                stiff case.  For a complex stiff ODE system in which f is not
                analytic, ZVODE is likely to have convergence failures, and
                for this problem one should instead use DVODE on the equivalent
                real system (in the real and imaginary parts of y).

        "lsoda"

            Real-valued Variable-coefficient Ordinary Differential Equation
            solver, with fixed-leading-coefficient implementation. It provides
            automatic method switching between implicit Adams method (for
            non-stiff problems) and a method based on backward differentiation
            formulas (BDF) (for stiff problems).

            Source: http://www.netlib.org/odepack

            .. warning::

               This integrator is not re-entrant. You cannot have two `ode`
               instances using the "lsoda" integrator at the same time.

            This integrator accepts the following keyword arguments:

            - atol : float or sequence
              absolute tolerance for solution
            - rtol : float or sequence
              relative tolerance for solution
            - lband : None or int
            - rband : None or int
              Jacobian band width, jac[i,j] != 0 for i-lband <= j <= i+rband.
              Setting these requires your jac routine to return the jacobian
              in packed format, jac_packed[i-j+lband, j] = jac[i,j].
            - with_jacobian : bool
              Whether to use the jacobian
            - nsteps : int
              Maximum number of (internally defined) steps allowed during one
              call to the solver.
            - first_step : float
            - min_step : float
            - max_step : float
              Limits for the step sizes used by the integrator.
            - max_order_ns : int
              Maximum order used in the nonstiff case (default 12).
            - max_order_s : int
              Maximum order used in the stiff case (default 5).
            - max_hnil : int
              Maximum number of messages reporting too small step size (t+h=t)
              (default 0)
            - ixpr : int
              Whether to generate extra printing at method switches. Default is
              False.

        "dopri5"

            This is an explicit Runge-Kutta method of order (4)5 due to Dormand
            and Prince (with adaptive step-size control and dense output).

            Authors:

                E. Hairer and G. Wanner
                Universite de Geneve, Dept. de Mathematiques
                CH-1211 Geneve 24, Switzerland
                e-mail: ernst.hairer@math.unige.ch,
                        gerhard.wanner@math.unige.ch

            This code is described in [HNW93]_.

            This integrator accepts the following keyword arguments:

            - atol : float or sequence
              absolute tolerance for solution
            - rtol : float or sequence
              relative tolerance for solution
            - nsteps : int
              Maximum number of (internally defined) steps allowed during one
              call to the solver.
            - first_step : float
            - max_step : float
            - safety : float
              Safety factor on new step selection (default 0.9)
            - ifactor : float
            - dfactor : float
              Maximum factor to increase/decrease step size by in one step
            - beta : float
              Beta parameter for stabilised step size control.
            - verbosity : int
              Switch for printing messages (< 0 for no messages).

        "dop853"

            This is an explicit Runge-Kutta method of order 8(5,3) due to
            Dormand and Prince (with adaptive step-size control and dense
            output).

            Options and references the same as "dopri5".

        Returns
        -------
            solution: array_like (float)
                Simulated solution trajectory.

        """
        # select the integrator
        self.ode.set_integrator(integrator, **kwargs)

        # pass the model parameters as additional args
        if self.args is not None:
            self.ode.set_f_params(*self.args)
            self.ode.set_jac_params(*self.args)

        # set the initial condition
        self.ode.set_initial_value(y0, t0)

        # create a storage container for the trajectory
        solution = np.hstack((t0, y0))

        # generate a solution trajectory
        while self.ode.successful():

            self.ode.integrate(self.ode.t + h, step, relax)
            current_step = np.hstack((self.ode.t, self.ode.y))
            solution = np.vstack((solution, current_step))

            # check terminal conditions
            if (g is not None) and (g(self.ode.t, self.ode.y, *self.args) < tol):
                break

            elif (T is not None) and (h > 0) and (self.ode.t >= T):
                break

            elif (T is not None) and (h < 0) and (self.ode.t <= T):
                break

            else:
                continue

        return solution

    def interpolate(self, traj, ti, k=3, der=0, ext=0):
        """
        Parametric B-spline interpolation in N-dimensions.

        Parameters
        ----------
            traj : array_like (float)
                Solution trajectory providing the data points for constructing
                the B-spline representation.
            ti : array_like (float)
                Array of values for the independent variable at which to
                interpolate the value of the B-spline.
            k : int, optional(default=3)
                Degree of the desired B-spline. Degree must satisfy
                :math:`1 \le k \le 5`.
            der : int, optional(default=0)
                The order of derivative of the spline to compute (must be less
                than or equal to `k`).
            ext : int, optional(default=2) Controls the value of returned
                elements for outside the original knot sequence provided by
                traj. For extrapolation, set `ext=0`; `ext=1` returns zero;
                `ext=2` raises a `ValueError`.

        Returns
        -------
            interp_traj: array (float)
                The interpolated trajectory.

        """
        # array of parameter values
        u = traj[:, 0]

        # build list of input arrays
        n = traj.shape[1]
        x = [traj[:, i] for i in range(1, n)]

        # construct the B-spline representation (s=0 forces interpolation!)
        tck, t = interpolate.splprep(x, u=u, k=k, s=0)

        # evaluate the B-spline (returns a list)
        out = interpolate.splev(ti, tck, der, ext)

        # convert to a 2D array
        interp_traj = np.hstack((ti[:, np.newaxis], np.array(out).T))

        return interp_traj

    def residual(self, traj, ti, k=3, ext=2):
        """
        The residual is the difference between the derivative of the B-spline
        approximation of the solution trajectory and the right-hand side of the
        original ODE evaluated along the approximated solution trajectory.

        Parameters
        ----------
            traj : array_like (float)
                Solution trajectory providing the data points for constructing
                the B-spline representation.
            ti : array_like (float)
                Array of values for the independent variable at which to
                interpolate the value of the B-spline.
            k : int, optional(default=3)
                Degree of the desired B-spline. Degree must satisfy
                :math:`1 \le k \le 5`.
            ext : int, optional(default=2)
                Controls the value of returned elements for outside the
                original knot sequence provided by traj. For extrapolation, set
                `ext=0`; `ext=1` returns zero; `ext=2` raises a `ValueError`.

        Returns
        -------
            residual: array (float)
                Difference between the derivative of the B-spline approximation
                of the solution trajectory and the right-hand side of the ODE
                evaluated along the approximated solution trajectory.

        """
        interp_soln = self.interpolate(traj, ti, k, 0, ext)
        interp_deriv = self.interpolate(traj, ti, k, 1, ext)
        residual = interp_deriv - self.f(ti, interp_soln[:, 1:])
        return residual

    def compare_trajectories(self, traj1, traj2):
        """
        Return the element-wise difference between two trajectories.

        Parameters
        ----------
            traj1 : array_like (float, shape=(T, N+1))
                Array containing a solution trajectory.
            traj2 : array_like (float, shape=(T, N+1))
                Array containing a solution trajectory.

        Returns
        -------
            abs_diff: array_like (float)
                Array containing the element-wise difference between traj1 and
                traj2.

        """
        diff = traj1[:, 1:] - traj2[:, 1:]
        return diff

    def get_l2_errors(self, traj1, traj2):
        """
        Computes a measure of the difference between two trajectories using
        the :math: `L^2` norm.

        Parameters
        ----------
            traj1 : array_like (float)
                Array containing a solution trajectory.
            traj2 : array_like (float)
                Array containing a solution trajectory.

        Returns
        -------
            l2_error: float
                Measure of the total difference between two trajectories.

        """
        l2_error = np.sum(self.compare_trajectories(traj1, traj2)**2)**0.5
        return l2_error

    def get_maximal_errors(self, traj1, traj2):
        """
        Computes a measure of the distance between two trajectories using the
        :math:`L^{\infty}` norm.

        Parameters
        ----------
            traj1 : array_like (float)
                Array containing a solution trajectory.
            traj2 : array_like (float)
                Array containing a solution trajectory.

        Returns
        -------
            maximal_error: float
                Maximal difference between two trajectories.

        """
        maximal_error = np.max(np.abs(self.compare_trajectories(traj1, traj2)))
        return maximal_error
