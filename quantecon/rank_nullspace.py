# This file is not meant for public use and will be removed v0.8.0.
# Use the `quantecon` namespace for importing the objects
# included below.

import warnings
from . import _rank_nullspace


__all__ = ['rank_est', 'nullspace']


def __dir__():
    return __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(
                "`quantecon.rank_nullspace` is deprecated and has no attribute"
                f" '{name}'."
            )

    warnings.warn(f"Please use `{name}` from the `quantecon` namespace, the"
                  "`quantecon.rank_nullspace` namespace is deprecated. You can"
                  f" use following instead:\n `from quantecon import {name}`.",
                  category=DeprecationWarning, stacklevel=2)

    return getattr(_rank_nullspace, name)
