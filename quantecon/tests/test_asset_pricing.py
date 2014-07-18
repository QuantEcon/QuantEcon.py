'''
  Test Module for asset_pricing.py
'''

import unittest
import quantecon.asset_pricing as ap

class TestSuite1(unittest.TestCase):
  # - Test Case In Functions Here - #
    def setUp(self):
        self.a = 1
        self.b = 2

    def testOnePlusTwo(self):
        self.assertTrue(self.a + self.b == 3)




if __name__ == '__main__':
  suite1 = unittest.TestLoader().loadTestsFromTestCase(TestSuite1)
  unittest.TextTestRunner(verbosity=2).run(suite1)

