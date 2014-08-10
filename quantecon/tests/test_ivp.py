"""
Test suite for ivp.py

"""
import unittest

import numpy as np

from .. import ivp
from ..models import lotka_volterra


class LotkaVolterraTest(unittest.TestCase):

    def setUp(self):
        # specify some parameters (a, b, c, d)
        params = (1.0, 0.1, 1.5, 0.75)

        # define the initial value problem (IVP)
        self.ivp = ivp.IVP(f=lotka_volterra.f,
                           jac=lotka_volterra.jacobian,
                           args=params)

    def test_compute_residual(self):
        """Test integration and interpolation."""
        # initial condition
        t0, X0 = 0.0, np.array([10.0, 5.0])

        integrators = ['dopri5', 'dop853', 'vode', 'lsoda']

        for integrator in integrators:

            # tighten tolerances so tests don't file due to numerical issues
            tmp_numeric_traj = self.ivp.integrate(t0, X0, h=1e-2, T=15.0,
                                                  integrator=integrator,
                                                  atol=1e-14, rtol=1e-11)
            T = tmp_numeric_traj[:, 0][-1]
            tmp_grid_pts = np.linspace(t0, T, 1000)

            # used highest order B-spline interpolation available
            tmp_residual = self.ivp.compute_residual(tmp_numeric_traj,
                                                     tmp_grid_pts,
                                                     k=5)

            expected_residual = np.zeros((1000, 3))
            actual_residual = tmp_residual
            np.testing.assert_almost_equal(expected_residual, actual_residual)
