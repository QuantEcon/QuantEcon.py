"""
Filename: timing.py
Authors: Pablo Winant
Date: 10/16/14
Provides Matlab-like tic, tac and toc functions.
"""


class __Timer__:
    '''Computes elapsed time, between tic, tac, and toc.

    Methods
    -------
    tic :
        Resets timer.
    toc :
        Returns and prints time elapsed since last tic().
    tac :
        Returns and prints time elapsed since last
             tic(), tac() or toc() whichever occured last.
    '''

    start = None
    last = None

    def tic(self):
        """Resets timer."""

        import time

        t = time.time()
        self.start = t
        self.last = t

    def tac(self):
        """Returns and prints time elapsed since last tic()"""

        import time

        if self.start is None:
            raise Exception("tac() without tic()")

        t = time.time()
        elapsed = t-self.last
        self.last = t

        print("TAC: Elapsed: {} seconds.".format(elapsed))
        return elapsed

    def toc(self):
        """Returns and prints time elapsed since last
        tic() or tac() whichever occured last"""

        import time

        if self.start is None:
            raise Exception("toc() without tic()")

        t = time.time()
        self.last = t
        elapsed = t-self.start

        print("TOC: Elapsed: {} seconds.".format(elapsed))
        return elapsed

__timer__ = __Timer__()


def tic():
    """Saves time for future use with tac or toc."""
    return __timer__.tic()


def tac():
    """Prints and returns elapsed time since last tic, tac or toc."""
    return __timer__.tac()


def toc():
    """Prints and returns elapsed time since last tic, tac or toc."""
    return __timer__.toc()
