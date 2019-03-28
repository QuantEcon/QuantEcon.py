"""
Filename: test_fictplay.py

Tests for fictplay.py

"""

import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.stats import norm

from quantecon.game_theory import FictitiousPlay, StochasticFictitiousPlay


class Test_FictitiousPlay_DecreaingGain:

    def setUp(self):
        '''Setup a FictitiousPlay instance'''
        # symmetric 2x2 coordination game
        matching_pennies = [[(1, -1), (-1, 1)],
                            [(-1, 1), (1, -1)]]
        self.fp = FictitiousPlay(matching_pennies)

    def test_play(self):
        x = [np.asarray([1, 0]), np.asarray([0.5, 0.5])]
        assert_array_almost_equal(self.fp.play(init_actions=(0, 0)), x)

    def test_time_series(self):
        x = self.fp.time_series(ts_length=3, init_actions=(0, 0))
        assert_array_almost_equal(x[0], [[1, 0],
                                         [1, 0],
                                         [1, 0]])
        assert_array_almost_equal(x[1], [[1, 0],
                                         [1/2, 1/2],
                                         [1/3, 2/3]])


class Test_FictitiousPlay_ConstantGain:

    def setUp(self):
        matching_pennies = [[(1, -1), (-1, 1)],
                            [(-1, 1), (1, -1)]]
        self.fp = FictitiousPlay(matching_pennies, gain=0.1)

    def test_play(self):
        x = [np.asarray([1, 0]), np.asarray([0.9, 0.1])]
        assert_array_almost_equal(self.fp.play(init_actions=(0, 0)), x)

    def test_time_series(self):
        x = self.fp.time_series(ts_length=3, init_actions=(0, 0))
        assert_array_almost_equal(x[0], [[1, 0],
                                         [1, 0],
                                         [1, 0]])
        assert_array_almost_equal(x[1], [[1, 0],
                                         [0.9, 0.1],
                                         [0.81, 0.19]])


class Test_StochasticFictitiosuPlay_DecreaingGain:

    def setUp(self):
        matching_pennies = [[(1, -1), (-1, 1)],
                            [(-1, 1), (1, -1)]]
        distribution = norm()
        self.fp = StochasticFictitiousPlay(matching_pennies,
                                           distribution=distribution)

    def test_play(self):
        seed = 1234
        x = [self.fp.play(init_actions=(0, 0),
                          random_state=np.random.RandomState(seed))
             for i in range(2)]
        assert_array_almost_equal(x[0], x[1])

    def test_time_series(self):
        seed = 1234
        x = [self.fp.time_series(ts_length=3, init_actions=(0, 0),
                                 random_state=np.random.RandomState(seed))
             for i in range(2)]
        assert_array_almost_equal(x[0][0], x[1][0])
        assert_array_almost_equal(x[0][1], x[1][1])


class Test_StochasticFictitiosuPlay_ConstantGain:

    def setUp(self):
        matching_pennies = [[(1, -1), (-1, 1)],
                            [(-1, 1), (1, -1)]]
        distribution = norm()
        self.fp = StochasticFictitiousPlay(matching_pennies, gain=0.1,
                                           distribution=distribution)

    def test_play(self):
        seed = 1234
        x = [self.fp.play(init_actions=(0, 0),
                          random_state=np.random.RandomState(seed))
             for i in range(2)]
        assert_array_almost_equal(x[0], x[1])

    def test_time_series(self):
        seed = 1234
        x = [self.fp.time_series(ts_length=3, init_actions=(0, 0),
                                 random_state=np.random.RandomState(seed))
             for i in range(2)]
        assert_array_almost_equal(x[0][0], x[1][0])
        assert_array_almost_equal(x[0][1], x[1][1])


if __name__ == '__main__':
    import sys
    import nose

    argv = sys.argv[:]
    argv.append('--verbose')
    argv.append('--nocapture')
    nose.main(argv=argv, defaultTest=__file__)
