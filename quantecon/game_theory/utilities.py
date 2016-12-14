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
