"""
Test suite for ces.py

"""
import unittest
import ..ces


class CESTestSuite(unittest.TestCase):

    def test_output(self):
        """Test function for CES output."""
        raise NotImplementedError

    def test_marginal_product_capital(self):
        """Test function for CES marginal product of capital."""
        raise NotImplementedError

    def test_marginal_product_labor(self):
        """Test function for CES marginal product of labor."""
        raise NotImplementedError


if __name__ == '__main__':
    CESTest = unittest.TestLoader().loadTestsFromTestCase(CESTestSuite)
    unittest.TextTestRunner(verbosity=2).run(suite1)