"""
Tests for estspec.py

TODO: write tests that check accuracy of returns

"""
import unittest
import numpy as np
from quantecon import smooth, periodogram, ar_periodogram
from quantecon.tests.util import capture


x_20 = np.random.rand(20)
x_21 = np.random.rand(21)


class PeriodogramBase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if cls is PeriodogramBase:
            raise unittest.SkipTest("Skip PeriodogramBase tests" +
                                    " it's a base class")

    def test_func_w_shape_even_x(self):
        self.assertEqual(self.w_20.size, x_20.size // 2 + 1)

    def test_func_w_shape_odd_x(self):
        self.assertEqual(self.w_21.size, x_21.size // 2 + 1)

    def test_func_Iw_shape_even_x(self):
        self.assertEqual(self.Iw_20.size, x_20.size // 2 + 1)

    def test_func_Iw_shape_odd_x(self):
        self.assertEqual(self.Iw_21.size, x_21.size // 2 + 1)

    def test_func_w_Iw_same_shape(self):
        self.assertEqual(self.w_20.shape, self.Iw_20.shape)
        self.assertEqual(self.w_21.shape, self.Iw_21.shape)

    def test_func_I(self):
        pass


class TestPeriodogram(PeriodogramBase):

    @classmethod
    def setUpClass(cls):
        if cls is PeriodogramBase:
            raise unittest.SkipTest("Skip BaseTest tests, it's a base class")
        super(TestPeriodogram, cls).setUpClass()
        cls.window_length = 7
        cls.w_20, cls.Iw_20 = periodogram(x_20)
        cls.w_21, cls.Iw_21 = periodogram(x_21)
        cls.funcname = "periodogram"


class TestArPeriodogram(PeriodogramBase):

    @classmethod
    def setUpClass(cls):
        if cls is PeriodogramBase:
            raise unittest.SkipTest("Skip BaseTest tests, it's a base class")
        super(TestArPeriodogram, cls).setUpClass()
        cls.window_length = 7
        cls.w_20, cls.Iw_20 = ar_periodogram(x_20)
        cls.w_21, cls.Iw_21 = ar_periodogram(x_21)
        cls.funcname = "ar_periodogram"

    # I need to over-ride these b/c this function always has
    # w.size == x.size //2
    def test_func_w_shape_even_x(self):
        self.assertEqual(self.w_20.size, x_20.size // 2)

    def test_func_Iw_shape_even_x(self):
        self.assertEqual(self.Iw_20.size, x_20.size // 2)


class TestSmooth(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.x_20 = np.random.rand(20)
        cls.x_21 = np.random.rand(21)
        cls.window_length = 7

    def test_smooth(self):  # does smoothing smooth?
        pass

    def test_smooth_raise_long_window(self):
        "estspec: raise error if smooth(*a, window_len) too large"
        self.assertRaises(ValueError, smooth, self.x_20, window_len=25)

    def test_smooth_short_window_err(self):
        "estspec: raise error in smooth(*a, window_len) if window_len too small"
        self.assertRaises(ValueError, smooth, self.x_20, window_len=2)

    def test_smooth_default_hanning(self):
        "estspec: smooth defaults to hanning on unrecognized window"
        with capture(smooth, x=self.x_20, window="foobar") as output:
            self.assertRegexpMatches(output, "Defaulting")

    def test_smooth_window_len_must_be_odd(self):
        "estspec: smooth changes even window_len to odd"
        with capture(smooth, x=self.x_20, window_len=4) as output:
            self.assertRegexpMatches(output, "reset")
