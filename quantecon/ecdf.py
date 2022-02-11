# This file is not meant for public use and will be removed v0.6.0.
# Use the `quantecon` namespace for importing the objects
# included below.

import warnings
from . import _ecdf


__all__ = ['ECDF']


def __dir__():
    return __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(
                f"`quantecon.ecdf` is deprecated and has no attribute {name}"
            )

    warnings.warn(f"Please use `{name}` from the `quantecon` namespace, "
                  "the `quantecon.ecdf` namespace is deprecated.",
                  category=DeprecationWarning, stacklevel=2)

    return getattr(_ecdf, name)
