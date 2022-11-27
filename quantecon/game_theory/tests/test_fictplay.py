"""
Filename: test_fictplay.py

Tests for fictplay.py

"""
import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.stats import norm

from quantecon.game_theory import FictitiousPlay, StochasticFictitiousPlay


class TestFictitiousPlayDecreaingGain:

    def setup_method(self):
        '''Setup a FictitiousPlay instance'''
        # symmetric 2x2 coordination game
        matching_pennies = [[(1, -1), (-1, 1)],
                            [(-1, 1), (1, -1)]]
        self.fp = FictitiousPlay(matching_pennies)

    def test_play(self):
        x = (np.array([1, 0]), np.array([0.5, 0.5]))
        assert_array_almost_equal(self.fp.play(actions=(0, 0)), x)

    def test_time_series(self):
        x = self.fp.time_series(ts_length=3, init_actions=(0, 0))
        assert_array_almost_equal(x[0], [[1, 0],
                                         [1, 0],
                                         [1, 0]])
        assert_array_almost_equal(x[1], [[1, 0],
                                         [1/2, 1/2],
                                         [1/3, 2/3]])


class TestFictitiousPlayConstantGain:

    def setup_method(self):
        matching_pennies = [[(1, -1), (-1, 1)],
                            [(-1, 1), (1, -1)]]
        self.fp = FictitiousPlay(matching_pennies, gain=0.1)

    def test_play(self):
        x = (np.array([1, 0]), np.array([0.9, 0.1]))
        assert_array_almost_equal(self.fp.play(actions=(0, 0)), x)

    def test_time_series(self):
        x = self.fp.time_series(ts_length=3, init_actions=(0, 0))
        assert_array_almost_equal(x[0], [[1, 0],
                                         [1, 0],
                                         [1, 0]])
        assert_array_almost_equal(x[1], [[1, 0],
                                         [0.9, 0.1],
                                         [0.81, 0.19]])


class TestStochasticFictitiosuPlayDecreaingGain:

    def setup_method(self):
        matching_pennies = [[(1, -1), (-1, 1)],
                            [(-1, 1), (1, -1)]]
        distribution = norm()
        self.fp = StochasticFictitiousPlay(matching_pennies,
                                           distribution=distribution)

    def test_play(self):
        seed = 272733541340907175684079858751241831341
        x = [self.fp.play(actions=(0, 0),
                          random_state=np.random.default_rng(seed))
             for i in range(2)]
        assert_array_almost_equal(x[0], x[1])

    def test_time_series(self):
        seed = 226177486389088886197048956835604946950
        x = [self.fp.time_series(ts_length=3, init_actions=(0, 0),
                                 random_state=np.random.default_rng(seed))
             for i in range(2)]
        assert_array_almost_equal(x[0][0], x[1][0])
        assert_array_almost_equal(x[0][1], x[1][1])


class TestStochasticFictitiosuPlayConstantGain:

    def setup_method(self):
        matching_pennies = [[(1, -1), (-1, 1)],
                            [(-1, 1), (1, -1)]]
        distribution = norm()
        self.fp = StochasticFictitiousPlay(matching_pennies, gain=0.1,
                                           distribution=distribution)

    def test_play(self):
        seed = 271001177347704493622442691590340912076
        x = [self.fp.play(actions=(0, 0),
                          random_state=np.random.default_rng(seed))
             for i in range(2)]
        assert_array_almost_equal(x[0], x[1])

    def test_time_series(self):
        seed = 143773081180220547556482766921740826832
        x = [self.fp.time_series(ts_length=3, init_actions=(0, 0),
                                 random_state=np.random.default_rng(seed))
             for i in range(2)]
        assert_array_almost_equal(x[0][0], x[1][0])
        assert_array_almost_equal(x[0][1], x[1][1])
