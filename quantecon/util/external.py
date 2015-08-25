"""
External Module
===============

This module is an import location for flexible dependancies on external packages such as numba

Packages
--------
  1. numba

"""

import warnings

#-Setup Installed Indicator and Import JIT function for NUMBA-#

numba_installed = True
try:
    from numba import jit
except ImportError:
    numba_installed = False
    jit = None
    from .common_messages import numba_import_fail_message
    warnings.warn(numba_import_fail_message, UserWarning)
