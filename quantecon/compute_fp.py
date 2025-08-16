# This file is not meant for public use and will be removed v0.8.0.
# Use the `quantecon` namespace for importing the objects
# included below.

import warnings
from . import _compute_fp


__all__ = ['compute_fixed_point']


def __dir__():
    return __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(
                "`quantecon.compute_fp` is deprecated and has no attribute "
                f"'{name}'."
            )

    warnings.warn(f"Please use `{name}` from the `quantecon` namespace, "
                  "the `quantecon.compute_fp` namespace is deprecated. You "
                  "can use the following instead:\n "
                  f"`from quantecon import {name}`.",
                  category=DeprecationWarning, stacklevel=2)

    return getattr(_compute_fp, name)
