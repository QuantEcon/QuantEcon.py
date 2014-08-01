"""
Test suite for solow.py

"""
import unittest
from ..solow import *


class SolowTestSuite(unittest.TestCase):
    pass

if __name__ == '__main__':
    SolowTest = unittest.TestLoader().loadTestsFromTestCase(SolowTestSuite)
    unittest.TextTestRunner(verbosity=2).run(suite1)