"""
Utilities to Support Random State Infrastructure
"""

import numpy as np
import numbers

#-Random States-#

def check_random_state(seed):
    """
    Check the random state of a given seed.

    If seed is None, return the RandomState singleton used by np.random.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.

    Otherwise raise ValueError.

    .. Note
       ----
        1. This code was sourced from scikit-learn

    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)
