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
    verbose : bool, optional(default=True)
        If True, print timing results. If False, suppress printing of timing results.
        
    Attributes
    ----------
    elapsed : float
        The elapsed time in seconds. Available after exiting the context.
        
    Examples
    --------
    Basic usage:
    >>> with Timer():
    ...     # some code
    ...     pass
    0.00 seconds elapsed
    
    With custom message and precision:
    >>> with Timer("Computing results", precision=4):
    ...     # some code  
    ...     pass
    Computing results: 0.0001 seconds elapsed
    
    Store elapsed time for comparison:
    >>> timer = Timer(verbose=False)
    >>> with timer:
    ...     # some code
    ...     pass
    >>> print(f"Method took {timer.elapsed:.6f} seconds")
    Method took 0.000123 seconds
    """
    
    def __init__(self, message="", precision=2, unit="seconds", verbose=True):
        self.message = message
        self.precision = precision
        self.unit = unit.lower()
        self.verbose = verbose
        self.elapsed = None
        self._start_time = None
        
        # Validate unit
        valid_units = ["seconds", "milliseconds", "microseconds"]
        if self.unit not in valid_units:
            raise ValueError(f"unit must be one of {valid_units}")
    
    def __enter__(self):
        self._start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        self.elapsed = end_time - self._start_time
        
        if self.verbose:
            self._print_elapsed()
            
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


def timeit(func, runs=1, stats_only=False, verbose=True, results=False, **timer_kwargs):
    """
    Execute a function multiple times and collect timing statistics.
    
    This function provides a convenient way to time a function multiple times
    and get summary statistics, using the Timer context manager internally.
    
    Parameters
    ----------
    func : callable
        Function to execute multiple times. Function should take no arguments,
        or be a partial function or lambda with arguments already bound.
    runs : int, optional(default=1)
        Number of runs to execute. Must be a positive integer.
    stats_only : bool, optional(default=False)
        If True, only display summary statistics. If False, display 
        individual run times followed by summary statistics.
    verbose : bool, optional(default=True)
        If True, print nicely formatted timing output all at once at the end.
        If False, suppress all output.
    results : bool, optional(default=False)
        If True, return dictionary with timing results. If False, return None.
    **timer_kwargs
        Keyword arguments to pass to Timer (message, precision, unit, verbose).
        
    Returns
    -------
    dict or None
        If results=True, returns dictionary containing timing results with keys:
        - 'elapsed': list of elapsed times for each run
        - 'average': average elapsed time
        - 'minimum': minimum elapsed time  
        - 'maximum': maximum elapsed time
        If results=False, returns None.
        
    Examples
    --------
    Basic usage:
    >>> def slow_function():
    ...     time.sleep(0.01)
    >>> timeit(slow_function, runs=3)
    Run 1: 0.01 seconds
    Run 2: 0.01 seconds
    Run 3: 0.01 seconds
    Average: 0.01 seconds, Minimum: 0.01 seconds, Maximum: 0.01 seconds
    
    Summary only:
    >>> timeit(slow_function, runs=3, stats_only=True) 
    Average: 0.01 seconds, Minimum: 0.01 seconds, Maximum: 0.01 seconds
    
    With custom Timer options:
    >>> timeit(slow_function, runs=2, unit="milliseconds", precision=1)
    Run 1: 10.1 ms
    Run 2: 10.0 ms  
    Average: 10.1 ms, Minimum: 10.0 ms, Maximum: 10.1 ms
    
    Return results for further analysis:
    >>> results = timeit(slow_function, runs=2, results=True)
    >>> print(f"Average time: {results['average']:.4f} seconds")
    
    Quiet mode:
    >>> timeit(slow_function, runs=2, verbose=False)  # No output
    
    With function arguments using lambda:
    >>> add_func = lambda: expensive_computation(5, 10)
    >>> timeit(add_func, runs=2)
    """
    if not isinstance(runs, int) or runs < 1:
        raise ValueError("runs must be a positive integer")
    
    if not callable(func):
        raise ValueError("func must be callable")
    
    # Extract Timer parameters
    timer_params = {
        'message': timer_kwargs.pop('message', ''),
        'precision': timer_kwargs.pop('precision', 2),
        'unit': timer_kwargs.pop('unit', 'seconds'),
        'verbose': timer_kwargs.pop('verbose', True)  # Timer verbose parameter
    }
    
    # Warn about unused kwargs
    if timer_kwargs:
        raise ValueError(f"Unknown timer parameters: {list(timer_kwargs.keys())}")
    
    # Determine if we should show output
    show_output = verbose
    
    run_times = []
    output_lines = []  # Collect output lines for printing all at once
    
    # Execute the function multiple times
    for i in range(runs):
        # Always silence individual Timer output to avoid duplication with our run display
        individual_timer_params = timer_params.copy()
        individual_timer_params['verbose'] = False
            
        with Timer(**individual_timer_params) as timer:
            func()
        run_times.append(timer.elapsed)
        
        # Collect individual run output lines (but don't print yet)
        if show_output and not stats_only:
            # Convert to requested unit for display
            unit = timer_params['unit'].lower()
            precision = timer_params['precision']
            
            if unit == "milliseconds":
                elapsed_display = timer.elapsed * 1000
                unit_str = "ms"
            elif unit == "microseconds":
                elapsed_display = timer.elapsed * 1000000
                unit_str = "μs"
            else:  # seconds
                elapsed_display = timer.elapsed
                unit_str = "seconds"
                
            output_lines.append(f"Run {i + 1}: {elapsed_display:.{precision}f} {unit_str}")
    
    # Calculate statistics
    average = sum(run_times) / len(run_times)
    minimum = min(run_times)
    maximum = max(run_times)
    
    # Collect summary statistics output line (but don't print yet)
    if show_output:
        # Convert to requested unit for display
        unit = timer_params['unit'].lower()
        precision = timer_params['precision']
        
        if unit == "milliseconds":
            avg_display = average * 1000
            min_display = minimum * 1000
            max_display = maximum * 1000
            unit_str = "ms"
        elif unit == "microseconds":
            avg_display = average * 1000000
            min_display = minimum * 1000000
            max_display = maximum * 1000000
            unit_str = "μs"
        else:  # seconds
            avg_display = average
            min_display = minimum
            max_display = maximum
            unit_str = "seconds"
            
        summary_line = (f"Average: {avg_display:.{precision}f} {unit_str}, "
                       f"Minimum: {min_display:.{precision}f} {unit_str}, "
                       f"Maximum: {max_display:.{precision}f} {unit_str}")
        output_lines.append(summary_line)
    
    # Print all output lines at once
    if show_output and output_lines:
        print('\n'.join(output_lines))
    
    # Return results only if requested
    if results:
        return {
            'elapsed': run_times,
            'average': average,
            'minimum': minimum,
            'maximum': maximum
        }
    else:
        return None


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
