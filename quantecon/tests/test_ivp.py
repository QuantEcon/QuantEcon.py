"""
Test suite for ivp.py

"""
import unittest
import ..ivp


class IVPTestSuite(unittest.TestCase):
    pass

if __name__ == '__main__':
    IVPTest = unittest.TestLoader().loadTestsFromTestCase(IVPTestSuite)
    unittest.TextTestRunner(verbosity=2).run(suite1)