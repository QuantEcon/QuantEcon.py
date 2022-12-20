# This file is not meant for public use and will be removed v0.8.0.
# Use the `quantecon` namespace for importing the objects
# included below.

import warnings
from . import _ce_util


__all__ = ['ckron', 'gridmake']


def __dir__():
    return __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(
                "`quantecon.ce_util` is deprecated and has no attribute "
                f"'{name}'."
            )

    warnings.warn(f"Please use `{name}` from the `quantecon` namespace, "
                  "the `quantecon.ce_util` namespace is deprecated. You can "
                  "use the following instead:\n "
                  f"`from quantecon import {name}`.",
                  category=DeprecationWarning, stacklevel=2)

    return getattr(_ce_util, name)
