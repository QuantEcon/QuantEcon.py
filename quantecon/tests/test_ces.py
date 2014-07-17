"""
Testing suite for ces.py

"""
import unittest
import ..ces


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