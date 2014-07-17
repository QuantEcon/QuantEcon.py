"""
Test suite for ces.py

"""
import unittest
import ..ces


class CESTestSuite(unittest.TestCase):
    pass

if __name__ == '__main__':
    CESTest = unittest.TestLoader().loadTestsFromTestCase(CESTestSuite)
    unittest.TextTestRunner(verbosity=2).run(suite1)