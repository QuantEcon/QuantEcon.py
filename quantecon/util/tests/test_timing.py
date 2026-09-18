"""
Tests for timing.py

"""

import time
import warnings
import pytest
from numpy.testing import assert_
from quantecon.util import tic, tac, toc, loop_timer, Timer, timeit
import quantecon as qe


# ``time.sleep(t)`` only guarantees the process is suspended for *at least*
# ``t`` seconds -- a busy or descheduled CI runner can make the measured time
# arbitrarily larger. The timing tests below therefore assert only a lower
# bound (that the timing utilities do not under-report the elapsed time) rather
# than an upper bound on wall-clock duration, which is inherently flaky on
# shared CI runners. See the discussion in GH #845.
TIMING_LOWER_BOUND = 0.9


def assert_at_least(measured, expected):
    """Assert ``measured`` seconds is not meaningfully below ``expected``."""
    lower_bound = expected * TIMING_LOWER_BOUND
    assert measured >= lower_bound, (
        f"measured {measured:.4f}s is below the lower bound {lower_bound:.4f}s "
        f"({TIMING_LOWER_BOUND:g} * expected {expected:.4f}s)")


# A measurement above this multiple of the expected sleep is far more likely a
# unit bug (``elapsed`` stored in ms/µs is ~1000x/1e6x larger) than scheduler
# overshoot, which is unbounded in theory but a few x at most under normal CI
# scheduling. The ceiling separates the two with negligible flake risk.
TIMING_UNIT_CEILING = 100


def assert_in_seconds(measured, expected):
    """Assert ``measured`` is in seconds (~``expected``), not a larger unit.

    Lower bound: the timer did not under-report. Upper bound: a ceiling not
    expected to be reached by an ``expected``-second sleep under normal CI
    scheduling, but which a unit leak into ``Timer.elapsed`` (ms/µs) always
    trips -- the invariant the unit tests guard.
    """
    lower_bound = expected * TIMING_LOWER_BOUND
    upper_bound = expected * TIMING_UNIT_CEILING
    assert lower_bound <= measured < upper_bound, (
        f"measured {measured:.4f}s is outside the seconds range "
        f"[{lower_bound:.4f}, {upper_bound:.4f}) around expected {expected:.4f}s "
        f"(did the display unit leak into elapsed?)")


class TestTicTacToc:
    def setup_method(self):
        self.h = 0.1
        self.digits = 10

    def test_timer(self):

        # tic/tac/toc are deprecated; silence the warnings for this
        # behavioural test (deprecation is asserted in TestDeprecation).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)

            tic()

            time.sleep(self.h)
            tm1 = tac()

            time.sleep(self.h)
            tm2 = tac()

            time.sleep(self.h)
            tm3 = toc()

        for actual, desired in zip([tm1, tm2, tm3],
                                   [self.h, self.h, self.h*3]):
            assert_at_least(actual, desired)

    def test_loop(self):

        def test_function_one_arg(n):
            return time.sleep(n)

        def test_function_two_arg(n, a):
            return time.sleep(n)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)

            test_one_arg = \
                loop_timer(5, test_function_one_arg, self.h, digits=10)
            test_two_arg = \
                loop_timer(5, test_function_two_arg, [self.h, 1], digits=10)

        for tm in test_one_arg:
            assert_at_least(tm, self.h)
        for tm in test_two_arg:
            assert_at_least(tm, self.h)

        for (average_time, average_of_best) in [test_one_arg, test_two_arg]:
            assert_(average_time >= average_of_best)


class TestDeprecation:
    """The Matlab-like tic/tac/toc/loop_timer functions are deprecated."""

    def test_tic_warns(self):
        with pytest.warns(DeprecationWarning, match="tic"):
            tic()

    def test_tac_warns(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            tic()
        with pytest.warns(DeprecationWarning, match="tac"):
            tac(verbose=False)

    def test_toc_warns(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            tic()
        with pytest.warns(DeprecationWarning, match="toc"):
            toc(verbose=False)

    def test_loop_timer_warns(self):
        def noop():
            pass

        with pytest.warns(DeprecationWarning, match="loop_timer"):
            loop_timer(2, noop, verbose=False)

    def test_loop_timer_warns_only_once(self):
        """`loop_timer` must not emit nested tic/tac/toc warnings per run."""
        def noop():
            pass

        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always", DeprecationWarning)
            loop_timer(3, noop, verbose=False)

        deprecations = [w for w in records
                        if issubclass(w.category, DeprecationWarning)]
        assert len(deprecations) == 1

    def test_warning_points_to_caller(self):
        """`stacklevel` should attribute the warning to the user's call site."""
        with pytest.warns(DeprecationWarning) as records:
            tic()  # noqa: this line is the expected warning source
        assert records[0].filename == __file__


class TestTimer:
    def setup_method(self):
        self.sleep_time = 0.1

    def test_basic_timer(self):
        """Test basic Timer context manager functionality."""
        timer = Timer(verbose=False)
        
        with timer:
            time.sleep(self.sleep_time)
            
        # Check that elapsed time was recorded
        assert timer.elapsed is not None
        assert_at_least(timer.elapsed, self.sleep_time)
        
    def test_timer_return_value(self):
        """Test that Timer returns self for variable assignment."""
        with Timer(verbose=False) as timer:
            time.sleep(self.sleep_time)
            
        assert timer.elapsed is not None
        assert_at_least(timer.elapsed, self.sleep_time)
        
    def test_timer_units(self):
        """Test different time units."""
        # Test seconds (default)
        with Timer(verbose=False) as timer_sec:
            time.sleep(self.sleep_time)
        expected_sec = self.sleep_time
        assert_in_seconds(timer_sec.elapsed, expected_sec)
        
        # Timer always stores elapsed time in seconds regardless of display unit
        with Timer(unit="milliseconds", verbose=False) as timer_ms:
            time.sleep(self.sleep_time)
        assert_in_seconds(timer_ms.elapsed, expected_sec)
        
        with Timer(unit="microseconds", verbose=False) as timer_us:
            time.sleep(self.sleep_time)
        assert_in_seconds(timer_us.elapsed, expected_sec)
        
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
        with Timer(precision=0, verbose=False) as timer0:
            time.sleep(self.sleep_time)
        with Timer(precision=6, verbose=False) as timer6:
            time.sleep(self.sleep_time)
            
        assert timer0.elapsed is not None
        assert timer6.elapsed is not None
        
    def test_timer_message(self):
        """Test custom message parameter (output format tested manually)."""
        with Timer(message="Test operation", verbose=False) as timer:
            time.sleep(self.sleep_time)
            
        assert timer.elapsed is not None
        
    def test_timer_verbose_mode(self):
        """Test verbose mode controls output."""
        # This mainly tests that verbose=False doesn't crash
        # Output suppression is hard to test automatically
        with Timer(verbose=False) as timer:
            time.sleep(self.sleep_time)
            
        assert timer.elapsed is not None
        
    def test_timer_exception_handling(self):
        """Test that Timer works correctly even when exceptions occur."""
        timer = Timer(verbose=False)
        
        try:
            with timer:
                time.sleep(self.sleep_time)
                raise ValueError("Test exception")
        except ValueError:
            pass  # Expected
            
        # Timer should still record elapsed time
        assert timer.elapsed is not None
        assert_at_least(timer.elapsed, self.sleep_time)

    def test_timeit_basic(self):
        """Test basic timeit functionality."""
        def test_func():
            time.sleep(self.sleep_time)
        
        result = timeit(test_func, runs=3, verbose=False, results=True)
        
        # Check that we have results
        assert 'elapsed' in result
        assert 'average' in result
        assert 'minimum' in result  
        assert 'maximum' in result
        assert isinstance(result['elapsed'], list)
        assert len(result['elapsed']) == 3
        
        # Check timing accuracy
        for run_time in result['elapsed']:
            assert_at_least(run_time, self.sleep_time)
            
        assert_at_least(result['average'], self.sleep_time)
        assert result['minimum'] <= result['average'] <= result['maximum']

    def test_timeit_lambda_function(self):
        """Test timeit with lambda functions for arguments."""
        def test_func_with_args(sleep_time, multiplier=1):
            time.sleep(sleep_time * multiplier)
        
        # Use lambda to bind arguments
        func_with_args = lambda: test_func_with_args(self.sleep_time, 0.5)
        result = timeit(func_with_args, runs=2, verbose=False, results=True)
        
        # Check results
        assert len(result['elapsed']) == 2
        for run_time in result['elapsed']:
            assert_at_least(run_time, self.sleep_time * 0.5)

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
            
        result = timeit(test_func, runs=1, verbose=False, results=True)
        
        assert len(result['elapsed']) == 1
        assert result['average'] == result['elapsed'][0]
        assert result['minimum'] == result['elapsed'][0]
        assert result['maximum'] == result['elapsed'][0]

    def test_timeit_different_units(self):
        """Test timeit with different time units."""
        def test_func():
            time.sleep(self.sleep_time)

        # Test milliseconds (verbose=False to avoid output during tests)
        result_ms = timeit(test_func, runs=2, unit="milliseconds", verbose=False, results=True)
        assert len(result_ms['elapsed']) == 2
        
        # Test microseconds
        result_us = timeit(test_func, runs=2, unit="microseconds", verbose=False, results=True) 
        assert len(result_us['elapsed']) == 2
        
        # All results should be in seconds regardless of display unit
        for run_time in result_ms['elapsed']:
            assert_in_seconds(run_time, self.sleep_time)
        for run_time in result_us['elapsed']:
            assert_in_seconds(run_time, self.sleep_time)

    def test_timeit_stats_only(self):
        """Test timeit with stats_only option."""
        def test_func():
            time.sleep(self.sleep_time)

        # This test is mainly to ensure stats_only doesn't crash
        result = timeit(test_func, runs=2, stats_only=True, verbose=False, results=True)
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

    def test_timeit_verbose_parameter(self):
        """Test verbose parameter controls output."""
        def test_func():
            time.sleep(self.sleep_time)
        
        # Test verbose=True (default) - this test can't easily verify output
        # but at least checks it doesn't crash
        result1 = timeit(test_func, runs=2, results=True)
        assert result1 is not None
        assert len(result1['elapsed']) == 2
        
        # Test verbose=False - should suppress output
        result2 = timeit(test_func, runs=2, verbose=False, results=True)
        assert result2 is not None
        assert len(result2['elapsed']) == 2

    def test_timeit_results_parameter(self):
        """Test results parameter controls return value."""
        def test_func():
            time.sleep(self.sleep_time)
        
        # Test results=False (default) - should return None
        result1 = timeit(test_func, runs=2, verbose=False)
        assert result1 is None
        
        # Test results=True - should return timing data
        result2 = timeit(test_func, runs=2, verbose=False, results=True)
        assert result2 is not None
        assert 'elapsed' in result2
        assert len(result2['elapsed']) == 2


class TestGlobalPrecision:
    """Test the new global precision control functionality."""
    
    def setup_method(self):
        """Save original precision and restore after each test."""
        self.original_precision = qe.timings.float_precision()
        
    def teardown_method(self):
        """Restore original precision after each test."""
        qe.timings.float_precision(self.original_precision)
        
    def test_default_precision_is_4(self):
        """Test that default precision is now 4."""
        # Reset to ensure we test the true default
        qe.timings.float_precision(4)
        assert qe.timings.float_precision() == 4
        
    def test_float_precision_get_set(self):
        """Test getting and setting precision."""
        # Test setting various precisions
        for precision in [0, 1, 2, 3, 4, 5, 6, 10]:
            qe.timings.float_precision(precision)
            assert qe.timings.float_precision() == precision
            
    def test_float_precision_validation(self):
        """Test that float_precision validates input."""
        # Test invalid inputs
        try:
            qe.timings.float_precision(-1)
            assert False, "Should raise ValueError for negative precision"
        except ValueError as e:
            assert "non-negative integer" in str(e)
            
        try:
            qe.timings.float_precision("4")
            assert False, "Should raise ValueError for string input"
        except ValueError as e:
            assert "non-negative integer" in str(e)
            
        try:
            qe.timings.float_precision(4.5)
            assert False, "Should raise ValueError for float input"
        except ValueError as e:
            assert "non-negative integer" in str(e)
            
    def test_timer_uses_global_precision(self):
        """Test that Timer class uses global precision by default."""
        # Set global precision
        qe.timings.float_precision(6)
        
        # Create timer without explicit precision
        timer = Timer(verbose=False)
        assert timer.precision == 6
        
        # Test with different global precision
        qe.timings.float_precision(2)
        timer2 = Timer(verbose=False)
        assert timer2.precision == 2
        
    def test_timer_explicit_precision_overrides_global(self):
        """Test that explicit precision overrides global setting."""
        qe.timings.float_precision(6)
        
        # Explicit precision should override global
        timer = Timer(precision=3, verbose=False)
        assert timer.precision == 3
        
    def test_tac_toc_keep_original_defaults(self):
        """Test that tac/toc functions maintain original default (digits=2)."""
        # These functions are deprecated and should maintain original behavior
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)

            tic()
            time.sleep(0.01)

            # These should use digits=2 by default, not global precision
            result_tac = tac(verbose=False)  # Uses default digits=2
            result_toc = toc(verbose=False)  # Uses default digits=2

        # Just verify they work without error
        assert result_tac > 0
        assert result_toc > 0
        
    def test_loop_timer_keeps_original_default(self):
        """Test that loop_timer maintains original default (digits=2)."""
        def test_func():
            time.sleep(0.001)
            
        # Should use digits=2 by default, not global precision
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = loop_timer(2, test_func, verbose=False)
        assert len(result) == 2  # Returns (average_time, average_of_best)
        
    def test_timeit_uses_global_precision(self):
        """Test that timeit function uses global precision by default."""
        def test_func():
            time.sleep(0.001)
            
        qe.timings.float_precision(6)
        
        # Should use global precision without error
        result = timeit(test_func, runs=2, verbose=False, results=True)
        assert 'elapsed' in result
        assert len(result['elapsed']) == 2

