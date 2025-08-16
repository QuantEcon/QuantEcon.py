"""
Tests for timing.py

"""

import time
from numpy.testing import assert_allclose, assert_
from quantecon.util import tic, tac, toc, loop_timer, Timer


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


class TestTimer:
    def setup_method(self):
        self.sleep_time = 0.1

    def test_basic_timer(self):
        """Test basic Timer context manager functionality."""
        timer = Timer(silent=True)
        
        with timer:
            time.sleep(self.sleep_time)
            
        # Check that elapsed time was recorded
        assert timer.elapsed is not None
        assert_allclose(timer.elapsed, self.sleep_time, atol=0.05, rtol=2)
        
    def test_timer_return_value(self):
        """Test that Timer returns self for variable assignment."""
        with Timer(silent=True) as timer:
            time.sleep(self.sleep_time)
            
        assert timer.elapsed is not None
        assert_allclose(timer.elapsed, self.sleep_time, atol=0.05, rtol=2)
        
    def test_timer_units(self):
        """Test different time units."""
        # Test seconds (default)
        with Timer(silent=True) as timer_sec:
            time.sleep(self.sleep_time)
        expected_sec = self.sleep_time
        assert_allclose(timer_sec.elapsed, expected_sec, atol=0.05, rtol=2)
        
        # Timer always stores elapsed time in seconds regardless of display unit
        with Timer(unit="milliseconds", silent=True) as timer_ms:
            time.sleep(self.sleep_time)
        assert_allclose(timer_ms.elapsed, expected_sec, atol=0.05, rtol=2)
        
        with Timer(unit="microseconds", silent=True) as timer_us:
            time.sleep(self.sleep_time)
        assert_allclose(timer_us.elapsed, expected_sec, atol=0.05, rtol=2)
        
    def test_invalid_unit(self):
        """Test that invalid units raise ValueError."""
        try:
            Timer(unit="invalid")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "unit must be one of" in str(e)
            
    def test_timer_precision(self):
        """Test that precision parameter is accepted (output format tested manually)."""
        # Just verify it doesn't crash with different precision values
        with Timer(precision=0, silent=True) as timer0:
            time.sleep(self.sleep_time)
        with Timer(precision=6, silent=True) as timer6:
            time.sleep(self.sleep_time)
            
        assert timer0.elapsed is not None
        assert timer6.elapsed is not None
        
    def test_timer_message(self):
        """Test custom message parameter (output format tested manually)."""
        with Timer(message="Test operation", silent=True) as timer:
            time.sleep(self.sleep_time)
            
        assert timer.elapsed is not None
        
    def test_timer_silent_mode(self):
        """Test silent mode suppresses output."""
        # This mainly tests that silent=True doesn't crash
        # Output suppression is hard to test automatically
        with Timer(silent=True) as timer:
            time.sleep(self.sleep_time)
            
        assert timer.elapsed is not None
        
    def test_timer_exception_handling(self):
        """Test that Timer works correctly even when exceptions occur."""
        timer = Timer(silent=True)
        
        try:
            with timer:
                time.sleep(self.sleep_time)
                raise ValueError("Test exception")
        except ValueError:
            pass  # Expected
            
        # Timer should still record elapsed time
        assert timer.elapsed is not None
        assert_allclose(timer.elapsed, self.sleep_time, atol=0.05, rtol=2)
