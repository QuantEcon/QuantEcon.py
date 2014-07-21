"""
Test suite for ivp.py

"""
import unittest

from numpy import testing

from ..ivp import *
from ..solow import *


class IVPTestSuite(unittest.TestCase):
    """Base class for ivp.py module tests."""

    def setUp(self):
        raise NotImplementedError

    def test_integrate(self):
        """Test ODE integration."""
        raise NotImplementedError

    def test_interpolate(self):
        """Test B_spline interpolation."""
        raise NotImplementedError


class SolowModelCase(IVPTestSuite):

    def setUp(self):
        # specify some parameters (g, n, s, alpha, delta, sigma)
        params = (0.02, 0.02, 0.15, 0.33, 0.04, 1.0)

        # define the initial value problem (IVP)
        self.ivp = IVP(f=ces_k_dot, jac=ces_jacobian, args=params)

    def test_integrate(self):
        """Test ODE integration."""
        t0 = 0.0
        k0 = 0.5

        traj = self.ivp.integrate(t0, k0, h=1.0, T=10.0, integrator='dopri5')

        expected_sol = cobb_douglas_analytic_solution(traj[:0], k0, *self.ivp.args)
        actual_sol = traj[:, 1]

        testing.assert_almost_equal(expected_sol, actual_sol)


class SpenceModelCase(IVPTestSuite):
    pass


class PredatorPreyModelCase(IVPTestSuite):
    pass


if __name__ == '__main__':
    SolowTest = unittest.TestLoader().loadTestsFromTestCase(SolowModelCase)
    unittest.TextTestRunner(verbosity=2).run(SolowTest)