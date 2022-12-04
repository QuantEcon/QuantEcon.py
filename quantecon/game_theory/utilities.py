"""
Utility routines for the game_theory submodule

"""
import numbers
import numpy as np


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


# _copy_action_to, _copy_action_profile_to

def _copy_action_to(dst, src):
    """
    Copy the pure action (int) or mixed action (array_like) in `src` to
    the empty ndarray `dst`.

    Parameters
    ----------
    dst : ndarray(float, ndim=1)

    src : scalar(int) or array_like(float, ndim=1)

    """
    if isinstance(src, numbers.Integral):  # pure action
        dst[:] = 0
        dst[src] = 1
    else:  # mixed action
        np.copyto(dst, src)


def _copy_action_profile_to(dst, src):
    """
    Copy the pure actions (int) or mixed actions (array_like) in the
    N-array_like `src` to the empty ndarrays in the N-array_like `dst`.

    Parameters
    ----------
    dst : array_like(ndarray(float, ndim=1))

    src : array_like(int or array_like(float, ndim=1))

    """
    N = len(dst)
    for i in range(N):
        _copy_action_to(dst[i], src[i])
