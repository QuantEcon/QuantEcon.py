"""
Provides Matlab-like tic, tac and toc functions.

"""
import time
import numpy as np



class __Timer__:
    """Computes elapsed time, between tic, tac, and toc.

    Methods
    -------
    tic :
        Resets timer.
    toc :
        Returns and prints time elapsed since last tic().
    tac :
        Returns and prints time elapsed since last
             tic(), tac() or toc() whichever occured last.
    loop_timer :
        Returns and prints the total and average time elapsed for n runs
        of a given function.

    """
    start = None
    last = None

    def tic(self):
        """
        Save time for future use with `tac()` or `toc()`.
        
        Returns
        -------
        None
            This function doesn't return a value.
        """
        t = time.time()
        self.start = t
        self.last = t

    def tac(self, verbose=True, digits=2):
        """
        Return and print elapsed time since last `tic()`, `tac()`, or
        `toc()`.

        Parameters
        ----------
        verbose : bool, optional(default=True)
            If True, then prints time.

        digits : scalar(int), optional(default=2)
            Number of digits printed for time elapsed.

        Returns
        -------
        elapsed : scalar(float)
            Time elapsed since last `tic()`, `tac()`, or `toc()`.

        """
        if self.start is None:
            raise Exception("tac() without tic()")

        t = time.time()
        elapsed = t-self.last
        self.last = t

        if verbose:
            m, s = divmod(elapsed, 60)
            h, m = divmod(m, 60)
            print("TAC: Elapsed: %d:%02d:%0d.%0*d" %
                  (h, m, s, digits, (s % 1)*(10**digits)))

        return elapsed

    def toc(self, verbose=True, digits=2):
        """
        Return and print time elapsed since last `tic()`.

        Parameters
        ----------
        verbose : bool, optional(default=True)
            If True, then prints time.

        digits : scalar(int), optional(default=2)
            Number of digits printed for time elapsed.

        Returns
        -------
        elapsed : scalar(float)
            Time elapsed since last `tic()`.

        """
        if self.start is None:
            raise Exception("toc() without tic()")

        t = time.time()
        self.last = t
        elapsed = t-self.start

        if verbose:
            m, s = divmod(elapsed, 60)
            h, m = divmod(m, 60)
            print("TOC: Elapsed: %d:%02d:%0d.%0*d" %
                  (h, m, s, digits, (s % 1)*(10**digits)))

        return elapsed

    def loop_timer(self, n, function, args=None, verbose=True, digits=2,
                   best_of=3):
        """
        Return and print the total and average time elapsed for n runs
        of function.

        Parameters
        ----------
        n : scalar(int)
            Number of runs.

        function : function
            Function to be timed.

        args : list, optional(default=None)
            Arguments of the function.

        verbose : bool, optional(default=True)
            If True, then prints average time.

        digits : scalar(int), optional(default=2)
            Number of digits printed for time elapsed.

        best_of : scalar(int), optional(default=3)
            Average time over best_of runs.

        Returns
        -------
        average_time : scalar(float)
            Average time elapsed for n runs of function.

        average_of_best : scalar(float)
            Average of best_of times for n runs of function.

        """
        tic()
        all_times = np.empty(n)
        for run in range(n):
            if hasattr(args, '__iter__'):
                function(*args)
            elif args is None:
                function()
            else:
                function(args)
            all_times[run] = tac(verbose=False, digits=digits)

        elapsed = toc(verbose=False, digits=digits)

        m, s = divmod(elapsed, 60)
        h, m = divmod(m, 60)

        print("Total run time: %d:%02d:%0d.%0*d" %
              (h, m, s, digits, (s % 1)*(10**digits)))

        average_time = all_times.mean()
        average_of_best = np.sort(all_times)[:best_of].mean()

        if verbose:
            m, s = divmod(average_time, 60)
            h, m = divmod(m, 60)
            print("Average time for %d runs: %d:%02d:%0d.%0*d" %
                  (n, h, m, s, digits, (s % 1)*(10**digits)))
            m, s = divmod(average_of_best, 60)
            h, m = divmod(m, 60)
            print("Average of %d best times: %d:%02d:%0d.%0*d" %
                  (best_of, h, m, s, digits, (s % 1)*(10**digits)))

        return average_time, average_of_best


__timer__ = __Timer__()


class Timer:
    """
    A context manager for timing code execution.
    
    This provides a modern context manager approach to timing, allowing
    patterns like `with Timer():` instead of manual tic/toc calls.
    
    Parameters
    ----------
    message : str, optional(default="")
        Custom message to display with timing results.
    precision : int, optional(default=2)
        Number of decimal places to display for seconds.
    unit : str, optional(default="seconds") 
        Unit to display timing in. Options: "seconds", "milliseconds", "microseconds"
    silent : bool, optional(default=False)
        If True, suppress printing of timing results.
    runs : int, optional(default=1)
        Number of runs to execute. If > 1, enables multiple runs mode.
        
    Attributes
    ----------
    elapsed : float or list
        For single run (runs=1): elapsed time in seconds.
        For multiple runs (runs>1): list of elapsed times for each run.
    minimum : float
        Minimum elapsed time (only available when runs > 1).
    maximum : float  
        Maximum elapsed time (only available when runs > 1).
    average : float
        Average elapsed time (only available when runs > 1).
        
    Examples
    --------
    Basic usage (single run):
    >>> with Timer():
    ...     # some code
    ...     pass
    0.00 seconds elapsed
    
    Multiple runs with callable:
    >>> def my_function():
    ...     time.sleep(0.01)
    >>> timer = Timer(runs=3, silent=True)
    >>> timer.timeit(my_function)
    >>> print(f"Average: {timer.average:.4f}s")
    Average: 0.0101s
    
    Multiple runs with callable:
    >>> def my_function():
    ...     time.sleep(0.01)
    >>> timer = Timer(runs=3)
    >>> timer.timeit(my_function)
    Run 1/3: 0.01 seconds
    Run 2/3: 0.01 seconds  
    Run 3/3: 0.01 seconds
    Average: 0.01 seconds, Min: 0.01 seconds, Max: 0.01 seconds
    """
    
    def __init__(self, message="", precision=2, unit="seconds", silent=False, runs=1):
        self.message = message
        self.precision = precision
        self.unit = unit.lower()
        self.silent = silent
        self.runs = runs
        self.elapsed = None
        self.minimum = None
        self.maximum = None
        self.average = None
        self._start_time = None
        
        # Validate unit
        valid_units = ["seconds", "milliseconds", "microseconds"]
        if self.unit not in valid_units:
            raise ValueError(f"unit must be one of {valid_units}")
        
        # Validate runs
        if not isinstance(runs, int) or runs < 1:
            raise ValueError("runs must be a positive integer")
    
    def __enter__(self):
        self._start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.runs == 1:
            # Single run mode - record elapsed time
            end_time = time.time()
            self.elapsed = end_time - self._start_time
            
            if not self.silent:
                self._print_elapsed()
        else:
            # Multiple runs mode - context manager not supported, only timeit()
            raise RuntimeError("Context manager usage is only supported for single runs (runs=1). For multiple runs, use the timeit() method.")
    
    def timeit(self, func, *args, **kwargs):
        """
        Execute a callable multiple times and collect timing statistics.
        Only available when runs > 1.
        
        Parameters
        ----------
        func : callable
            Function to execute multiple times
        *args, **kwargs
            Arguments to pass to func
            
        Returns
        -------
        None
            Results are stored in elapsed, minimum, maximum, average attributes
        """
        if self.runs == 1:
            raise RuntimeError("timeit() is only available when runs > 1. Use context manager for single runs.")
        
        run_times = []
        for i in range(self.runs):
            start_time = time.time()
            func(*args, **kwargs)
            end_time = time.time()
            run_time = end_time - start_time
            run_times.append(run_time)
            
            if not self.silent:
                self._print_single_run(i + 1, run_time)
        
        # Store results
        self.elapsed = run_times
        self.minimum = min(run_times)
        self.maximum = max(run_times)
        self.average = sum(run_times) / len(run_times)
        
        if not self.silent:
            self._print_multiple_runs_summary()

    def _print_single_run(self, run_number, run_time):
        """Print timing for a single run in multiple runs mode."""
        # Convert to requested unit
        if self.unit == "milliseconds":
            elapsed_display = run_time * 1000
            unit_str = "ms"
        elif self.unit == "microseconds":
            elapsed_display = run_time * 1000000
            unit_str = "μs"
        else:  # seconds
            elapsed_display = run_time
            unit_str = "seconds"
            
        print(f"Run {run_number}/{self.runs}: {elapsed_display:.{self.precision}f} {unit_str}")
    
    def _print_multiple_runs_summary(self):
        """Print summary statistics for multiple runs."""
        # Convert to requested unit
        if self.unit == "milliseconds":
            avg_display = self.average * 1000
            min_display = self.minimum * 1000
            max_display = self.maximum * 1000
            unit_str = "ms"
        elif self.unit == "microseconds":
            avg_display = self.average * 1000000
            min_display = self.minimum * 1000000
            max_display = self.maximum * 1000000
            unit_str = "μs"
        else:  # seconds
            avg_display = self.average
            min_display = self.minimum
            max_display = self.maximum
            unit_str = "seconds"
            
        prefix = f"{self.message}: " if self.message else ""
        print(f"{prefix}Average: {avg_display:.{self.precision}f} {unit_str}, "
              f"Min: {min_display:.{self.precision}f} {unit_str}, "
              f"Max: {max_display:.{self.precision}f} {unit_str}")
            
    def _print_elapsed(self):
        """Print the elapsed time with appropriate formatting."""
        # Convert to requested unit
        if self.unit == "milliseconds":
            elapsed_display = self.elapsed * 1000
            unit_str = "ms"
        elif self.unit == "microseconds":
            elapsed_display = self.elapsed * 1000000
            unit_str = "μs"
        else:  # seconds
            elapsed_display = self.elapsed
            unit_str = "seconds"
            
        # Format the message
        if self.message:
            prefix = f"{self.message}: "
        else:
            prefix = ""
            
        print(f"{prefix}{elapsed_display:.{self.precision}f} {unit_str} elapsed")


def tic():
    return __timer__.tic()


def tac(verbose=True, digits=2):
    return __timer__.tac(verbose, digits)


def toc(verbose=True, digits=2):
    return __timer__.toc(verbose, digits)


def loop_timer(n, function, args=None, verbose=True, digits=2, best_of=3):
    return __timer__.loop_timer(n, function, args, verbose, digits, best_of)


# Set docstring
_names = ['tic', 'tac', 'toc', 'loop_timer']
_funcs = [eval(name) for name in _names]
_methods = [getattr(__Timer__, name) for name in _names]
for _func, _method in zip(_funcs, _methods):
    _func.__doc__ = _method.__doc__
