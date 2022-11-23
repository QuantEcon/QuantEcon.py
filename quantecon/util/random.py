"""
Utilities to Support Random State Infrastructure
"""

import numpy as np
import numbers

# To be called as util.rng_integers
from scipy._lib._util import rng_integers  # noqa: F401


# Random States

def check_random_state(seed):
    """
    Turn `seed` into a `np.random.RandomState` instance.

    Parameters
    ----------
    seed : None, int, `np.random.RandomState`, or `np.random.Generator`
        If None, the `np.random.RandomState` singleton is returned. If
        `seed` is an int, a new ``RandomState`` instance seeded with
        `seed` is returned. If `seed` is already a `RandomState` or
        `Generator` instance, then that instance is returned.

    Returns
    -------
    `np.random.RandomState` or `np.random.Generator`
        Random number generator.

    Notes
    -----
    This code was originally sourced from scikit-learn.

    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, (np.random.RandomState, np.random.Generator)):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)
