"""
Testing suite for ces.py

"""
import unittest

from numpy import testing
import sympy as sp

from ..ces import *

class CESTestSuite(unittest.TestCase):
    """Base class for ces.py module tests."""

    def setUp(self):
        raise NotImplementedError

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

    def setUp(self):
        """Use SymPy to construct some functions for use in testing."""
        sp.var('K, A, L, alpha, beta, sigma')

        # compute symbolic expressions
        _sp_output = K**alpha * (A * L)**beta
        _sp_mpk = sp.diff(_sp_output, K)
        _sp_mpl = sp.diff(_sp_output, L)
        _sp_elasticity_YK = (K / _sp_output) * _sp_mpk
        _sp_elasticity_YL = (L / _sp_output) * _sp_mpl

        # wrap SymPy expressions into callable NumPy funcs
        args = (K, A, L, alpha, beta, sigma)
        self.np_output = sp.lambdify(args, _sp_output, 'numpy')
        self.np_mpk = sp.lambdify(args, _sp_mpk, 'numpy')
        self.np_mpl = sp.lambdify(args, _sp_mpl, 'numpy')
        self.np_elasticity_YK = sp.lambdify(args, _sp_elasticity_YK, 'numpy')
        self.np_elasticity_YL = sp.lambdify(args, _sp_elasticity_YL, 'numpy')

        # create a grid of input values
        T = 10
        eps = 1e-2
        capital_pts = np.linspace(eps, 10.0, T)
        labor_pts = capital_pts
        technology_pts = np.exp(capital_pts)
        self.input_grid = np.meshgrid(capital_pts, technology_pts, labor_pts)

        # create a grid of parameter values
        alpha_vals = np.linspace(eps, 1 - eps, T)
        beta_vals = np.linspace(eps, 1 - eps, T)
        sigma_vals = np.ones(T)
        self.parameter_grid = np.meshgrid(alpha_vals, beta_vals, sigma_vals)

    def test_marginal_product_capital(self):
        """Test Cobb-Douglas marginal product of capital."""
        # unpack grids
        capital, technology, labor = self.input_grid
        alpha, beta, sigma = self.parameter_grid

        # vectorize original function to accept parameter arrays
        test_mpk = np.vectorize(marginal_product_capital)

        # conduct the test
        expected_mpk = self.np_mpk(capital, technology, labor,
                                   alpha, beta, sigma)
        actual_mpk = test_mpk(capital, technology, labor,
                              alpha, beta, sigma)
        testing.assert_almost_equal(expected_mpk, actual_mpk)

    def test_marginal_product_labor(self):
        """Test Cobb-Douglas marginal product of labor."""
        # unpack grids
        capital, technology, labor = self.input_grid
        alpha, beta, sigma = self.parameter_grid

        # vectorize original function to accept parameter arrays
        test_mpl = np.vectorize(marginal_product_labor)

        # conduct the test
        expected_mpl = self.np_mpl(capital, technology, labor,
                                   alpha, beta, sigma)
        actual_mpl = test_mpl(capital, technology, labor,
                              alpha, beta, sigma)
        testing.assert_almost_equal(expected_mpl, actual_mpl)

    def test_output(self):
        """Test Cobb-Douglas output."""
        # unpack grids
        capital, technology, labor = self.input_grid
        alpha, beta, sigma = self.parameter_grid

        # vectorize original function to accept parameter arrays
        test_output = np.vectorize(output)

        # conduct the test
        expected_output = self.np_output(capital, technology, labor,
                                         alpha, beta, sigma)
        actual_output = test_output(capital, technology, labor,
                                    alpha, beta, sigma)
        testing.assert_almost_equal(expected_output, actual_output)

    def test_output_elasticity_capital(self):
        """Test Cobb-Douglas elasticity of output with respect to capital."""
        # unpack grids
        capital, technology, labor = self.input_grid
        alpha, beta, sigma = self.parameter_grid

        # vectorize original function to accept parameter arrays
        test_elasticity = np.vectorize(output_elasticity_capital)

        # conduct the test
        expected_elasticity = self.np_elasticity_YK(capital, technology, labor,
                                                    alpha, beta, sigma)
        actual_elasticity = test_elasticity(capital, technology, labor,
                                            alpha, beta, sigma)
        testing.assert_almost_equal(expected_elasticity, actual_elasticity)

    def test_output_elasticity_labor(self):
        """Test Cobb-Douglas elasticity of output with respect to labor."""
        # unpack grids
        capital, technology, labor = self.input_grid
        alpha, beta, sigma = self.parameter_grid

        # vectorize original function to accept parameter arrays
        test_elasticity = np.vectorize(output_elasticity_labor)

        # conduct the test
        expected_elasticity = self.np_elasticity_YL(capital, technology, labor,
                                                    alpha, beta, sigma)
        actual_elasticity = test_elasticity(capital, technology, labor,
                                            alpha, beta, sigma)
        testing.assert_almost_equal(expected_elasticity, actual_elasticity)


class LeontiefCase(CESTestSuite):

    def setUp(self):
        """Create a grid of inputs and parameters over which to iterate."""
        # create a grid of input values
        T = 10
        eps = 1e-2
        capital_pts = np.linspace(eps, 10.0, T)
        labor_pts = capital_pts
        technology_pts = np.exp(capital_pts)
        self.input_grid = np.meshgrid(capital_pts, technology_pts, labor_pts)

        # create a grid of parameter values
        alpha_vals = np.linspace(eps, 1 - eps, T)
        beta_vals = np.linspace(eps, 1 - eps, T)
        sigma_vals = np.zeros(T)
        self.parameter_grid = np.meshgrid(alpha_vals, beta_vals, sigma_vals)

    def test_marginal_product_capital(self):
        """Test Leontief  marginal product of capital."""
        # unpack grids
        capital, technology, labor = self.input_grid
        alpha, beta, sigma = self.parameter_grid

        # vectorize original function to accept parameter arrays
        test_mpk = np.vectorize(marginal_product_capital)

        # conduct the test
        expected_mpk = np.where(alpha * capital < beta * technology * labor,
                                alpha, 0.0)
        actual_mpk = test_mpk(capital, technology, labor,
                              alpha, beta, sigma)
        testing.assert_almost_equal(expected_mpk, actual_mpk)

    def test_marginal_product_labor(self):
        """Test Leontief  marginal product of labor."""
        # unpack grids
        capital, technology, labor = self.input_grid
        alpha, beta, sigma = self.parameter_grid

        # vectorize original function to accept parameter arrays
        test_mpl = np.vectorize(marginal_product_labor)

        # conduct the test
        expected_mpl = np.where(beta * technology * labor < alpha * capital,
                                beta * technology, 0.0)
        actual_mpl = test_mpl(capital, technology, labor,
                              alpha, beta, sigma)
        testing.assert_almost_equal(expected_mpl, actual_mpl)

    def test_output(self):
        """Test Leontief  output."""
        # unpack grids
        capital, technology, labor = self.input_grid
        alpha, beta, sigma = self.parameter_grid

        # vectorize original function to accept parameter arrays
        test_output = np.vectorize(output)

        # conduct the test
        expected_output = np.minimum(alpha * capital, beta * technology * labor)
        actual_output = test_output(capital, technology, labor,
                                    alpha, beta, sigma)
        testing.assert_almost_equal(expected_output, actual_output)

    def test_output_elasticity_capital(self):
        """Test Leontief  elasticity of output with respect to capital."""
        # unpack grids
        capital, technology, labor = self.input_grid
        alpha, beta, sigma = self.parameter_grid

        # vectorize original function to accept parameter arrays
        test_elasticity = np.vectorize(output_elasticity_capital)

        # conduct the test
        expected_elasticity = self.np_elasticity_YK(capital, technology, labor,
                                                    alpha, beta, sigma)
        actual_elasticity = test_elasticity(capital, technology, labor,
                                            alpha, beta, sigma)
        testing.assert_almost_equal(expected_elasticity, actual_elasticity)

    def test_output_elasticity_labor(self):
        """Test Leontief elasticity of output with respect to labor."""
        # unpack grids
        capital, technology, labor = self.input_grid
        alpha, beta, sigma = self.parameter_grid

        # vectorize original function to accept parameter arrays
        test_elasticity = np.vectorize(output_elasticity_labor)

        # conduct the test
        expected_elasticity = self.np_elasticity_YL(capital, technology, labor,
                                                    alpha, beta, sigma)
        actual_elasticity = test_elasticity(capital, technology, labor,
                                            alpha, beta, sigma)
        testing.assert_almost_equal(expected_elasticity, actual_elasticity)

class GeneralCESCase(CESTestSuite):

    def setUp(self):
        """Use SymPy to construct some functions for use in testing."""
        sp.var('K, A, L, alpha, beta, sigma')
        rho = (sigma - 1) / sigma

        # compute symbolic expressions
        _sp_output = (alpha * K**rho + beta * (A * L)**rho)**(1 / rho)
        _sp_mpk = sp.diff(_sp_output, K)
        _sp_mpl = sp.diff(_sp_output, L)
        _sp_elasticity_YK = (K / _sp_output) * _sp_mpk
        _sp_elasticity_YL = (L / _sp_output) * _sp_mpl

        # wrap SymPy expressions into callable NumPy funcs
        args = (K, A, L, alpha, beta, sigma)
        self.np_output = sp.lambdify(args, _sp_output, 'numpy')
        self.np_mpk = sp.lambdify(args, _sp_mpk, 'numpy')
        self.np_mpl = sp.lambdify(args, _sp_mpl, 'numpy')
        self.np_elasticity_YK = sp.lambdify(args, _sp_elasticity_YK, 'numpy')
        self.np_elasticity_YL = sp.lambdify(args, _sp_elasticity_YL, 'numpy')

        # create a grid of input values
        T = 10
        eps = 1e-2
        capital_pts = np.linspace(eps, 10.0, T)
        labor_pts = capital_pts
        technology_pts = np.exp(capital_pts)
        self.input_grid = np.meshgrid(capital_pts, technology_pts, labor_pts)

        # create a grid of parameter values
        alpha_vals = np.linspace(eps, 1 - eps, T)
        beta_vals = np.linspace(eps, 1 - eps, T)
        sigma_vals = np.logspace(2, 2, T)
        self.parameter_grid = np.meshgrid(alpha_vals, beta_vals, sigma_vals)

    def test_marginal_product_capital(self):
        """Test CES marginal product of capital."""
        # unpack grids
        capital, technology, labor = self.input_grid
        alpha, beta, sigma = self.parameter_grid

        # vectorize original function to accept parameter arrays
        test_mpk = np.vectorize(marginal_product_capital)

        # conduct the test
        expected_mpk = self.np_mpk(capital, technology, labor,
                                   alpha, beta, sigma)
        actual_mpk = test_mpk(capital, technology, labor,
                              alpha, beta, sigma)
        testing.assert_almost_equal(expected_mpk, actual_mpk)

    def test_marginal_product_labor(self):
        """Test CES marginal product of labor."""
        # unpack grids
        capital, technology, labor = self.input_grid
        alpha, beta, sigma = self.parameter_grid

        # vectorize original function to accept parameter arrays
        test_mpl = np.vectorize(marginal_product_labor)

        # conduct the test
        expected_mpl = self.np_mpl(capital, technology, labor,
                                   alpha, beta, sigma)
        actual_mpl = test_mpl(capital, technology, labor,
                              alpha, beta, sigma)
        testing.assert_almost_equal(expected_mpl, actual_mpl)

    def test_output(self):
        """Test CES output."""
        # unpack grids
        capital, technology, labor = self.input_grid
        alpha, beta, sigma = self.parameter_grid

        # vectorize original function to accept parameter arrays
        test_output = np.vectorize(output)

        # conduct the test
        expected_output = self.np_output(capital, technology, labor,
                                         alpha, beta, sigma)
        actual_output = test_output(capital, technology, labor,
                                    alpha, beta, sigma)
        testing.assert_almost_equal(expected_output, actual_output)

    def test_output_elasticity_capital(self):
        """Test CES elasticity of output with respect to capital."""
        # unpack grids
        capital, technology, labor = self.input_grid
        alpha, beta, sigma = self.parameter_grid

        # vectorize original function to accept parameter arrays
        test_elasticity = np.vectorize(output_elasticity_capital)

        # conduct the test
        expected_elasticity = self.np_elasticity_YK(capital, technology, labor,
                                                    alpha, beta, sigma)
        actual_elasticity = test_elasticity(capital, technology, labor,
                                            alpha, beta, sigma)
        testing.assert_almost_equal(expected_elasticity, actual_elasticity)

    def test_output_elasticity_labor(self):
        """Test CES elasticity of output with respect to labor."""
        # unpack grids
        capital, technology, labor = self.input_grid
        alpha, beta, sigma = self.parameter_grid

        # vectorize original function to accept parameter arrays
        test_elasticity = np.vectorize(output_elasticity_labor)

        # conduct the test
        expected_elasticity = self.np_elasticity_YL(capital, technology, labor,
                                                    alpha, beta, sigma)
        actual_elasticity = test_elasticity(capital, technology, labor,
                                            alpha, beta, sigma)
        testing.assert_almost_equal(expected_elasticity, actual_elasticity)


if __name__ == '__main__':
    test_loader = unittest.TestLoader()

    cobb_douglas = test_loader.loadTestsFromTestCase(CobbDouglasCase)
    leontief = test_loader.loadTestsFromTestCase(LeontiefCase)
    general_ces = test_loader.loadTestsFromTestCase(GeneralCESCase)

    unittest.TextTestRunner(verbosity=2).run(cobb_douglas)
    unittest.TextTestRunner(verbosity=2).run(leontief)
    unittest.TextTestRunner(verbosity=2).run(general_ces)