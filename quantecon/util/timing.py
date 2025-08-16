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
    >>> timer = Timer(silent=True)
    >>> with timer:
    ...     # some code
    ...     pass
    >>> print(f"Method took {timer.elapsed:.6f} seconds")
    Method took 0.000123 seconds
    """
    
    def __init__(self, message="", precision=2, unit="seconds", silent=False):
        self.message = message
        self.precision = precision
        self.unit = unit.lower()
        self.silent = silent
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
        
        if not self.silent:
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
