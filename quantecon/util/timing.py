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
