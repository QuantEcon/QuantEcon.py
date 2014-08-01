"""
Test suite for ivp.py

"""
import unittest

from numpy import testing

from ..ivp import *
from ..solow import *


class SolowModelCase(unittest.TestCase):

    def setUp(self):
        # specify some parameters (g, n, s, alpha, delta, sigma)
        params = (0.02, 0.02, 0.15, 0.33, 0.04, 1.0)

        # define the initial value problem (IVP)
        self.ivp = IVP(f=ces_k_dot, jac=ces_jacobian, args=params)

    def test_compute_residual(self):
        """Test accuracy of IVP solvers."""
        # analytic test case using Cobb-Douglas production technology.
        t0 = 0.0
        k0 = 0.5

        # tighten tolerances so tests don't file due to numerical issues
        kwargs = {'atol': 1e-12, 'rtol': 1e-9}

        integrators = ['dopri5', 'dop853', 'vode', 'lsoda']

        for integrator in integrators:

            tmp_numeric_traj = self.ivp.integrate(t0, k0, h=1e-1, T=100.0,
                                                  integrator=integrator,
                                                  **kwargs)
            T = tmp_numeric_traj[:, 0][-1]
            tmp_grid_pts = np.linspace(t0, T, 1000)

            tmp_residual = self.ivp.compute_residual(tmp_numeric_traj,
                                                     tmp_grid_pts,
                                                     k=3)
            expected_residual = np.zeros((1000, 2))
            actual_residual = tmp_residual

            testing.assert_almost_equal(expected_residual, actual_residual)

    def test_integrate(self):
        """Test ODE integration."""
        # analytic test case using Cobb-Douglas production technology.
        t0 = 0.0
        k0 = 0.5

        # tighten tolerances so tests don't file due to numerical issues
        kwargs = {'atol': 1e-12, 'rtol': 1e-9}

        integrators = ['dopri5', 'dop853', 'vode', 'lsoda']

        for integrator in integrators:

            tmp_numeric_traj = self.ivp.integrate(t0, k0, h=1e-1, T=100.0,
                                                  integrator=integrator,
                                                  **kwargs)

            tmp_grid_pts = tmp_numeric_traj[:, 0]
            tmp_analytic_traj = cobb_douglas_analytic_solution(tmp_grid_pts,
                                                               k0,
                                                               *self.ivp.args)

            expected_sol = tmp_analytic_traj[:, 1]
            actual_sol = tmp_numeric_traj[:, 1]

            testing.assert_almost_equal(expected_sol, actual_sol)

    def test_interpolate(self):
        """Test B-spline interpolation."""
        # analytic test case using Cobb-Douglas production technology.
        t0 = 0.0
        k0 = 0.5

        # tighten tolerances so tests don't file due to numerical issues
        kwargs = {'atol': 1e-12, 'rtol': 1e-9}

        integrators = ['dopri5', 'dop853', 'vode', 'lsoda']

        for integrator in integrators:

            tmp_numeric_traj = self.ivp.integrate(t0, k0, h=1e-1, T=100.0,
                                                  integrator=integrator,
                                                  **kwargs)
            T = tmp_numeric_traj[:, 0][-1]
            tmp_grid_pts = np.linspace(t0, T, 1000)

            tmp_interp_traj = self.ivp.interpolate(tmp_numeric_traj, tmp_grid_pts, k=3)
            tmp_analytic_traj = cobb_douglas_analytic_solution(tmp_grid_pts,
                                                               k0,
                                                               *self.ivp.args)

            expected_sol = tmp_analytic_traj[:, 1]
            actual_sol = tmp_interp_traj[:, 1]

            testing.assert_almost_equal(expected_sol, actual_sol)
