"""
Tests for timing.py

"""

import time
from numpy.testing import assert_allclose, assert_
from quantecon.util import tic, tac, toc, loop_timer, Timer, timeit


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

    def test_timeit_basic(self):
        """Test basic timeit functionality."""
        def test_func():
            time.sleep(self.sleep_time)
        
        result = timeit(test_func, runs=3, silent=True)
        
        # Check that we have results
        assert 'elapsed' in result
        assert 'average' in result
        assert 'minimum' in result  
        assert 'maximum' in result
        assert isinstance(result['elapsed'], list)
        assert len(result['elapsed']) == 3
        
        # Check timing accuracy
        for run_time in result['elapsed']:
            assert_allclose(run_time, self.sleep_time, atol=0.05, rtol=2)
            
        assert_allclose(result['average'], self.sleep_time, atol=0.05, rtol=2)
        assert result['minimum'] <= result['average'] <= result['maximum']

    def test_timeit_lambda_function(self):
        """Test timeit with lambda functions for arguments."""
        def test_func_with_args(sleep_time, multiplier=1):
            time.sleep(sleep_time * multiplier)
        
        # Use lambda to bind arguments
        func_with_args = lambda: test_func_with_args(self.sleep_time, 0.5)
        result = timeit(func_with_args, runs=2, silent=True)
        
        # Check results
        assert len(result['elapsed']) == 2
        for run_time in result['elapsed']:
            assert_allclose(run_time, self.sleep_time * 0.5, atol=0.05, rtol=2)

    def test_timeit_validation(self):
        """Test validation for timeit function."""
        def test_func():
            time.sleep(self.sleep_time)
            
        # Test invalid runs parameter
        try:
            timeit(test_func, runs=0)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "runs must be a positive integer" in str(e)
            
        try:
            timeit(test_func, runs=-1)
            assert False, "Should have raised ValueError"  
        except ValueError as e:
            assert "runs must be a positive integer" in str(e)
            
        # Test invalid function
        try:
            timeit("not a function", runs=1)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "func must be callable" in str(e)

    def test_timeit_single_run(self):
        """Test that timeit works with single run."""
        def test_func():
            time.sleep(self.sleep_time)
            
        result = timeit(test_func, runs=1, silent=True)
        
        assert len(result['elapsed']) == 1
        assert result['average'] == result['elapsed'][0]
        assert result['minimum'] == result['elapsed'][0]
        assert result['maximum'] == result['elapsed'][0]

    def test_timeit_different_units(self):
        """Test timeit with different time units."""
        def test_func():
            time.sleep(self.sleep_time)

        # Test milliseconds (silent mode to avoid output during tests)
        result_ms = timeit(test_func, runs=2, unit="milliseconds", silent=True)
        assert len(result_ms['elapsed']) == 2
        
        # Test microseconds
        result_us = timeit(test_func, runs=2, unit="microseconds", silent=True) 
        assert len(result_us['elapsed']) == 2
        
        # All results should be in seconds regardless of display unit
        for run_time in result_ms['elapsed']:
            assert_allclose(run_time, self.sleep_time, atol=0.05, rtol=2)
        for run_time in result_us['elapsed']:
            assert_allclose(run_time, self.sleep_time, atol=0.05, rtol=2)

    def test_timeit_stats_only(self):
        """Test timeit with stats_only option."""
        def test_func():
            time.sleep(self.sleep_time)

        # This test is mainly to ensure stats_only doesn't crash
        result = timeit(test_func, runs=2, stats_only=True, silent=True)
        assert len(result['elapsed']) == 2

    def test_timeit_invalid_timer_kwargs(self):
        """Test that invalid timer kwargs are rejected.""" 
        def test_func():
            time.sleep(self.sleep_time)
            
        try:
            timeit(test_func, runs=1, invalid_param="test")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Unknown timer parameters" in str(e)

