"""
Utility routines for the game_theory submodule

"""
class NashResult(dict):
    """
    Contain the information about the result of Nash equilibrium
    computation.

    Attributes
    ----------
    NE : tuple(ndarray(float, ndim=1))
        Computed Nash equilibrium.

    converged : bool
        Whether the routine has converged.

    num_iter : int
        Number of iterations.

    max_iter : int
        Maximum number of iterations.

    init : scalar or array_like
        Initial condition used.

    Notes
    -----
    This is sourced from `sicpy.optimize.OptimizeResult`.

    There may be additional attributes not listed above depending of the
    routine.

    """
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in sorted(self.items())])
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return self.keys()


class RGUtil:
    def frange(start, stop, step=1.):
        """
        Return evenly spaced floats within [start, stop].

        Parameters
        ----------
        start : scalar(float), optional(default=0.)
            Start of the interval.

        stop : scalar(float)
            End of the interval

        step : scalar(float), optional(default=1.)
            Spacing between values.

        Returns
        -------
        Generator object

        """
        i = 0.0
        x = float(start)  # Prevent yielding integers.
        x0 = x
        epsilon = step / 2.0
        yield x  # yield always first value
        while x + epsilon < stop:
            i += 1.0
            x = x0 + i * step
            yield x

    def unitcircle(npts):
        """
        Places `npts` equally spaced points along the 2 dimensional circle and 
        returns the points with x coordinates in first column and y coordinates
         in second column

        Parameters
        ----------
        npts : scalar(float)

        Returns
        -------
        ndarray(float, dim=2)

        """
        import math
        import numpy as np

        incr = 2 * math.pi / npts
        degrees = [x for x in RGUtil.frange(0, 2*math.pi, incr)]
        
        pts = np.empty((npts, 2))
        for i in range(npts):
            x = degrees[i]
            pts[i, 0] = math.cos(x)
            pts[i, 1] = math.sin(x)
        return pts