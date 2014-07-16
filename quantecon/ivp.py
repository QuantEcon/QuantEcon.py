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
            of endogenous variables with ``y.shape == (n,)``. The function `f`
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
        Generates solution trajectories of the model given some initial
        conditions.

        Arguments:

            t0:         (float) Initial condition for the independent variable.

            y0:         (float) Initial condition for the dependent variable.

            h:          (float) Step-size for computing the solution.

            T:          (int) Length of desired trajectory.

            g:          (callable) Function of the form g(t, vec, f_args) that
                        provides stopping conditions for the integration.
                        If specified, user must also specify a stopping
                        tolerance, tol. Default is None.

            tol:        (float) Stopping tolerance. On required if g is given.
                        Default is None.

            integrator: (str) Must be one of:

                        'forward_euler':    Basic implementation of Euler's
                                            method with fixed step size. See
                                            Judd (1998), Chapter 10, pg 341 for
                                            more detail.

                        'backward_euler':   Basic implementation of the
                                            implicit Euler method with a
                                            fixed step size.  See Judd (1998),
                                            Chapter 10, pg. 343 for more detail.

                        'trapezoidal_rule': Basic implementation of the
                                            trapezoidal rule with a fixed step
                                            size.  See Judd (1998), Chapter 10,
                                            pg. 344 for more detail.

                        'erk2':             Second-order explicit Runge-Kutta.

                        'erk3':             Third-order explicit Runge-Kutta.

                        'erk4':             Fourth-order explicit Runge-Kutta.

                        'erk5':             Fifth-order explicit Runge-Kutta.

                        'vode':             Real-valued Variable-coefficient ODE
                                            equation solver, with fixed leading
                                            coefficient implementation. It
                                            provides implicit Adams method (for
                                            non-stiff problems) and a method
                                            based on backward differentiation
                                            formulas (BDF) (for stiff problems).

                        'lsoda':            Real-valued Variable-coefficient ODE
                                            equation solver, with fixed leading
                                            coefficient implementation. It
                                            provides automatic method switching
                                            between implicit Adams method (for
                                            non-stiff problems) and a method
                                            based on backward differentiation
                                            formulas (BDF) (for stiff problems).

                        'dopri5':           Embedded explicit Runge-Kutta method
                                            with order 4(5). See Dormand and
                                            Prince (1980) for details.
                        'dop85':

                        See documentation for integrate.ode for more details and
                        references for 'vode', 'lsoda', 'dopri5', and 'dop85',
                        as well as the rest of the ODE solvers available via
                        ODEPACK.

            step:       (boolean) Allows access to internal steps for those
                        solvers that use adaptive step size routines. Currently
                        only 'vode', 'zvode', and 'lsoda' allow support step.
                        Default is False.

            relax:      (boolean) The following integrators support run_relax:
                        'vode', 'zvode', 'lsoda'. Default is False.

            **kwargs:   (dict) Dictionary of integrator specific keyword args.

        Returns:

           solution: (array-like) Simulated solution trajectory.

        """
        # select the integrator
        self.ode.set_integrator(integrator, **kwargs)

        # pass the model parameters as additional args
        if self.args != None:
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
            if g is not None and g(self.ode.t, self.ode.y, *self.args) < tol:
                break

            elif T is not None and h > 0 and self.ode.t >= T:
                break

            elif T is not None and h < 0 and self.ode.t <= T:
                break

            else:
                pass

        return solution

    def interpolate(self, traj, ti, k=3, der=0, ext=0):
        """
        Parameteric B-spline interpolation in N-dimensions.

        Arguments:

            traj: (array-like) Solution trajectory providing the data points for
                  constructing the B-spline representation.

            ti:   (array-like) Array of values for the independent variable at
                  which to interpolate the value of the B-spline.

            k:    (int) Degree of the desired B-spline. Degree must satsify
                  1 <= k <= 5. Default is k=3 for cubic B-spline interpolation.

            der:  (int) The order of derivative of the spline to compute
                  (must be less than or equal to k). Default is zero.

            ext: (int) Controls the value of returned elements for outside the
                 original knot sequence provided by traj. For extrapolation, set
                 ext=0; ext=1 returns zero; ext=2 raises a ValueError. Default
                 is to perform extrapolation.

        Returns:

            interp_traj: (array) The interpolated trajectory.

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

    def compare_trajectories(self, traj1, traj2):
        """
        Returns the absolute difference between two solution trajectories.

        Arguments:

            traj1: (array-like) (T,n+1) array containing a solution trajectory.
            traj2: (array-like) (T,n+1) array containing a solution trajectory.

        Returns:

            abs_diff: (array-like) (T,n) array of the element-wise absolute
                      difference between traj1 and traj2.
        """
        abs_diff = np.abs(traj1[:, 1:] - traj2[:, 1:])

        return abs_diff

    def get_L2_errors(self, traj1, traj2):
        """
        Computes a measure of the total difference between two trajectories
        using the L^2 norm.

        Arguments:

            traj1: (array-like) (T,n+1) array containing a solution trajectory.
            traj2: (array-like) (T,n+1) array containing a solution trajectory.

        Returns:

            L2_error: (float) Average difference between two trajectories.

        """
        L2_error = np.sum(self.compare_trajectories(traj1, traj2)**2)**0.5
        return L2_error

    def get_maximal_errors(self, traj1, traj2):
        """
        Computes a measure of the point-wise distance between two trajectories.

        Arguments:

            traj1: (array-like) (T,n+1) array containing a solution trajectory.
            traj2: (array-like) (T,n+1) array containing a solution trajectory.

        Returns:

            maximal_error: (float) Maximal difference between two trajectories.

        """
        maximal_error = np.max(self.compare_trajectories(traj1, traj2))
        return maximal_error
