"""
Testing suite for ces.py

"""
import unittest

from numpy import testing

import ..ces as ces


class CESTestSuite(unittest.TestCase):
    """Base class for ces.py module tests."""

    def test_marginal_product_capital(self):
        """Test CES marginal product of capital."""
        raise NotImplementedError

    def test_marginal_product_labor(self):
        """Test CES marginal product of labor."""
        raise NotImplementedError

    def test_output(self):
        """Test CES output."""
        raise NotImplementedError

    def test_output_elasticity_capital(self):
        """Test CES elasticity of output with respect to capital."""
        raise NotImplementedError

    def test_output_elasticity_labor(self):
        """Test CES elasticity of output with respect to capital."""
        raise NotImplementedError


class CobbDouglasCase(CESTestSuite):

    sigma = 1.0

    def test_marginal_product_capital(self):
        """Test CES marginal product of capital."""
        # inputs
        T = 100
        capital = np.repeat(4.0, T)
        techology = np.ones(T)
        labor = np.repeat(3.0, T)

        # CRTS test case
        alpha = 0.5
        beta = 1 - alpha
        expected_mpk = np.repeat(0.75, T)
        actual_mpk = ces.marginal_product_capital(capital, techology, labor,
                                                  alpha, beta, self.sigma)
        testing.assert_almost_equal(expected_mpk, actual_mpk)

    def test_marginal_product_labor(self):
        """Test CES marginal product of labor."""
        raise NotImplementedError

    def test_output(self):
        """Test CES output."""
        raise NotImplementedError

    def test_output_elasticity_capital(self):
        """Test CES elasticity of output with respect to capital."""
        raise NotImplementedError

    def test_output_elasticity_labor(self):
        """Test CES elasticity of output with respect to capital."""
        raise NotImplementedError


class LeontiefCase(CESTestSuite):

    sigma = 1e-6

    def test_marginal_product_capital(self):
        """Test CES marginal product of capital."""
        raise NotImplementedError

    def test_marginal_product_labor(self):
        """Test CES marginal product of labor."""
        raise NotImplementedError

    def test_output(self):
        """Test CES output."""
        raise NotImplementedError

    def test_output_elasticity_capital(self):
        """Test CES elasticity of output with respect to capital."""
        raise NotImplementedError

    def test_output_elasticity_labor(self):
        """Test CES elasticity of output with respect to capital."""
        raise NotImplementedError


class GeneralCESCase(CESTestSuite):

    def test_marginal_product_capital(self):
        """Test CES marginal product of capital."""
        raise NotImplementedError

    def test_marginal_product_labor(self):
        """Test CES marginal product of labor."""
        raise NotImplementedError

    def test_output(self):
        """Test CES output."""
        raise NotImplementedError

    def test_output_elasticity_capital(self):
        """Test CES elasticity of output with respect to capital."""
        raise NotImplementedError

    def test_output_elasticity_labor(self):
        """Test CES elasticity of output with respect to capital."""
        raise NotImplementedError


if __name__ == '__main__':
    test_loader = unittest.TestLoader()

    cobb_douglas = test_loader.loadTestsFromTestCase(CobbDouglasCase)
    leontief = test_loader.loadTestsFromTestCase(LeontiefCase)
    general_ces = test_loader.loadTestsFromTestCase(GeneralCESCase)

    unittest.TextTestRunner(verbosity=2).run(cobb_douglas)
    unittest.TextTestRunner(verbosity=2).run(leontief)
    unittest.TextTestRunner(verbosity=2).run(general_ces)