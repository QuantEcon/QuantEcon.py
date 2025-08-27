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

    def test_multiple_runs_basic(self):
        """Test basic multiple runs functionality."""
        def test_func():
            time.sleep(self.sleep_time)
        
        timer = Timer(runs=3, silent=True)
        timer.timeit(test_func)
        
        # Check that we have results
        assert timer.elapsed is not None
        assert isinstance(timer.elapsed, list)
        assert len(timer.elapsed) == 3
        assert timer.minimum is not None
        assert timer.maximum is not None
        assert timer.average is not None
        
        # Check timing accuracy
        for run_time in timer.elapsed:
            assert_allclose(run_time, self.sleep_time, atol=0.05, rtol=2)
        
        # Check statistics
        assert_allclose(timer.average, self.sleep_time, atol=0.05, rtol=2)
        assert timer.minimum <= timer.average <= timer.maximum
        
    def test_multiple_runs_with_args(self):
        """Test multiple runs with function arguments."""
        def test_func_with_args(sleep_time, multiplier=1):
            time.sleep(sleep_time * multiplier)
        
        timer = Timer(runs=2, silent=True)
        timer.timeit(test_func_with_args, self.sleep_time, multiplier=2)
        
        expected_time = self.sleep_time * 2
        assert len(timer.elapsed) == 2
        for run_time in timer.elapsed:
            assert_allclose(run_time, expected_time, atol=0.05, rtol=2)
    
    def test_multiple_runs_validation(self):
        """Test validation for multiple runs mode."""
        # Test invalid runs parameter
        try:
            Timer(runs=0)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "runs must be a positive integer" in str(e)
        
        try:
            Timer(runs=-1)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "runs must be a positive integer" in str(e)
        
        try:
            Timer(runs="invalid")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "runs must be a positive integer" in str(e)
    
    def test_timeit_single_run_error(self):
        """Test that timeit() raises error when runs=1."""
        timer = Timer(runs=1, silent=True)
        
        def dummy_func():
            pass
        
        try:
            timer.timeit(dummy_func)
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            assert "timeit() is only available when runs > 1" in str(e)
    
    def test_multiple_runs_different_units(self):
        """Test multiple runs with different time units."""
        def test_func():
            time.sleep(self.sleep_time)
        
        # Test milliseconds
        timer_ms = Timer(runs=2, unit="milliseconds", silent=True)
        timer_ms.timeit(test_func)
        
        # Times should still be stored in seconds internally
        for run_time in timer_ms.elapsed:
            assert_allclose(run_time, self.sleep_time, atol=0.05, rtol=2)
        
        # Test microseconds
        timer_us = Timer(runs=2, unit="microseconds", silent=True)
        timer_us.timeit(test_func)
        
        for run_time in timer_us.elapsed:
            assert_allclose(run_time, self.sleep_time, atol=0.05, rtol=2)
    
    def test_multiple_runs_message(self):
        """Test multiple runs with custom message."""
        def test_func():
            time.sleep(self.sleep_time)
        
        timer = Timer(runs=2, message="Test operation", silent=True)
        timer.timeit(test_func)
        
        assert len(timer.elapsed) == 2
        assert timer.average is not None
    
    def test_context_manager_multiple_runs_error(self):
        """Test that context manager usage raises error when runs > 1."""
        timer = Timer(runs=3, silent=True)
        
        try:
            with timer:
                time.sleep(self.sleep_time)
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            assert "Context manager usage is only supported for single runs" in str(e)

