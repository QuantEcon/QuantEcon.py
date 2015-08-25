"""
API for QuantEcon Utilities
"""

from .array import searchsorted
from .external import jit, numba_installed
from .random import check_random_state
from .timing import tic, tac, toc