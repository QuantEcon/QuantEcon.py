"""
Test suite for ivp.py

"""
import unittest

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
        """Test B_splien interpolation."""
        raise NotImplementedError


if __name__ == '__main__':
    IVPTest = unittest.TestLoader().loadTestsFromTestCase(IVPTestSuite)
    unittest.TextTestRunner(verbosity=2).run(suite1)