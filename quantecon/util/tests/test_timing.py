"""
Tests for timing.py

"""

import time
from numpy.testing import assert_allclose, assert_
from quantecon.util import tic, tac, toc, loop_timer


class TestTicTacToc:
    def setup_method(self):
        self.h = 0.1
        self.digits = 10

    def test_timer(self):

        tic()

        time.sleep(self.h)
        tm1 = tac()

        time.sleep(self.h)
        tm2 = tac()

        time.sleep(self.h)
        tm3 = toc()

        rtol = 2
        atol = 0.05

        for actual, desired in zip([tm1, tm2, tm3],
                                   [self.h, self.h, self.h*3]):
            assert_allclose(actual, desired, atol=atol, rtol=rtol)

    def test_loop(self):

        def test_function_one_arg(n):
            return time.sleep(n)

        def test_function_two_arg(n, a):
            return time.sleep(n)

        test_one_arg = \
            loop_timer(5, test_function_one_arg, self.h, digits=10)
        test_two_arg = \
            loop_timer(5, test_function_two_arg, [self.h, 1], digits=10)

        rtol = 2
        atol = 0.05

        for tm in test_one_arg:
            assert_allclose(tm, self.h, atol=atol, rtol=rtol)
        for tm in test_two_arg:
            assert_allclose(tm, self.h, atol=atol, rtol=rtol)

        for (average_time, average_of_best) in [test_one_arg, test_two_arg]:
            assert_(average_time >= average_of_best)
