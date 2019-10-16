"""
Tests for timing.py

"""

import time
from sys import platform
from numpy.testing import assert_allclose
from nose.tools import ok_
from quantecon.util import tic, tac, toc, loop_timer


class TestTicTacToc():
    def setUp(self):
        self.h = 0.1
        self.digits = 10

    def test_timer(self):
        if platform == 'darwin':
            # skip for darwin
            return

        tic()

        time.sleep(self.h)
        tm1 = tac()

        time.sleep(self.h)
        tm2 = tac()

        time.sleep(self.h)
        tm3 = toc()

        rtol = 0.1
        for actual, desired in zip([tm1, tm2, tm3],
                                   [self.h, self.h, self.h*3]):
            assert_allclose(actual, desired, rtol=rtol)

    def test_loop(self):
        if platform == 'darwin':
            # skip for darwin
            return

        def test_function_one_arg(n):
            return time.sleep(n)

        def test_function_two_arg(n, a):
            return time.sleep(n)

        test_one_arg = \
            loop_timer(5, test_function_one_arg, self.h, digits=10)
        test_two_arg = \
            loop_timer(5, test_function_two_arg, [self.h, 1], digits=10)
        for tm in test_one_arg:
            assert(abs(tm - self.h) < 0.01), tm
        for tm in test_two_arg:
            assert(abs(tm - self.h) < 0.01), tm

        for (average_time, average_of_best) in [test_one_arg, test_two_arg]:
            ok_(average_time >= average_of_best)


if __name__ == '__main__':
    import sys
    import nose

    argv = sys.argv[:]
    argv.append('--verbose')
    argv.append('--nocapture')
    nose.main(argv=argv, defaultTest=__file__)
