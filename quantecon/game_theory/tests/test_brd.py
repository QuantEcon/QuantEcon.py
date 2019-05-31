"""
Filename: test_brd.py

Tests for brd.py

"""
import numpy as np
from numpy.testing import assert_array_equal
from nose.tools import eq_

from quantecon.game_theory import BRD, KMR, SamplingBRD


class TestBRD:
    '''Test the methods of BRD'''

    def setUp(self):
        '''Setup a BRD instance'''
        # 2x2 coordination game with action 1 risk-dominant
        payoff_matrix = [[4, 0],
                         [3, 2]]
        self.N = 4  # 4 players
        self.brd = BRD(payoff_matrix, self.N)

    def test_time_series_1(self):
        assert_array_equal(
            self.brd.time_series(ts_length=3, init_action_dist=[4, 0]),
            [[4, 0],
             [4, 0],
             [4, 0]]
            )

    def test_time_series_2(self):
        seed = 1234
        x = [self.brd.time_series(ts_length=3, init_action_dist=[2, 2],
                                  random_state=np.random.RandomState(seed))
             for i in range(2)]
        assert_array_equal(x[0], x[1])


class TestKMR:
    '''Test the methods of KMR'''

    def setUp(self):
        payoff_matrix = [[4, 0],
                         [3, 2]]
        self.N = 4
        self.kmr = KMR(payoff_matrix, self.N)

    def test_time_series(self):
        seed = 1234
        x = [self.kmr.time_series(ts_length=3, init_action_dist=[2, 2],
                                  random_state=np.random.RandomState(seed))
             for i in range(2)]
        assert_array_equal(x[0], x[1])


class TestSamplingBRD:
    '''Test the methods of SamplingBRD'''

    def setUp(self):
        payoff_matrix = [[4, 0],
                         [3, 2]]
        self.N = 4
        self.sbrd = SamplingBRD(payoff_matrix, self.N)

    def test_time_series(self):
        seed = 1234
        x = [self.sbrd.time_series(ts_length=3, init_action_dist=[2, 2],
                                   random_state=np.random.RandomState(seed))
             for i in range(2)]
        assert_array_equal(x[0], x[1])


if __name__ == '__main__':
    import sys
    import nose

    argv = sys.argv[:]
    argv.append('--verbose')
    argv.append('--nocapture')
    nose.main(argv=argv, defaultTest=__file__)
