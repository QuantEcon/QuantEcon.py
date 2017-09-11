"""
Filename: timing.py
Authors: Pablo Winant
Tests for timing.py
"""

import time
from numpy.testing import assert_allclose
from nose.tools import eq_
from quantecon.util import tic, tac, toc, loop_timer


class TestTicTacToc():
    def setUp(self):
        self.h = 0.1 
        self.digits = 10
    
    def test_timer(self):
        tic()

        time.sleep(self.h)
        tm1 = float(tac())

        time.sleep(self.h)
        tm2 = float(tac())

        time.sleep(self.h)
        tm3 = float(toc())

        rtol = 0.1
        for actual, desired in zip([tm1, tm2, tm3], [self.h, self.h, self.h*3]):
            assert_allclose(actual, desired, rtol=rtol)

    def test_digits(self):
        tic()
        
        time.sleep(self.h)
        tm1 = tac(self.digits)

        time.sleep(self.h)
        tm2 = tac(self.digits)

        time.sleep(self.h)
        tm3 = toc(self.digits)
        
        eq_(len(str(tm1).split(".")[1]), 10)
        eq_(len(str(tm2).split(".")[1]), 10)
        eq_(len(str(tm3).split(".")[1]), 10)
        
    def test_output(self):
        tic()
        
        time.sleep(self.h)
        tm1 = tac(self.digits, True, False)

        time.sleep(self.h)
        tm2 = tac(self.digits, True, False)

        time.sleep(self.h)
        tm3 = toc(self.digits, True, False)
        
        eq_(tm1, None)
        eq_(tm2, None)
        eq_(tm3, None)
        
    def test_loop(self):
        def test_function_one_arg(n):
            return time.sleep(n)

        def test_function_two_arg(n, a):
            return time.sleep(n)

        test_one_arg = loop_timer(5, test_function_one_arg, self.h, 10)
        test_two_arg = loop_timer(5, test_function_two_arg, [self.h, 1], 10)
        for tm in test_one_arg:
            assert(abs(float(tm)-self.h)<0.01)
        for tm in test_two_arg:
            assert(abs(float(tm)-self.h)<0.01)

        
if __name__ == '__main__':
    import sys
    import nose

    argv = sys.argv[:]
    argv.append('--verbose')
    argv.append('--nocapture')
    nose.main(argv=argv, defaultTest=__file__)