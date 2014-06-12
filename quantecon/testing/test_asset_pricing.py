'''
  Test Module for asset_pricing.py
'''

import unittest
import ..asset_pricing as ap

class TestSuite1(unittest.TestCase):
  # - Test Case In Functions Here - #
  pass
  
if __name__ == '__main__':
  suite1 = unittest.TestLoader().loadTestsFromTestCase(TestSuite1)
  unittest.TextTestRunner(verbosity=2).run(suite1)

