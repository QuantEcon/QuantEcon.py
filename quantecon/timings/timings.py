"""
Global timing precision configuration for QuantEcon.py

This module provides global control over the precision used in timing outputs
across all timing functions in QuantEcon.
"""

# Global variable to store the current float precision
_DEFAULT_FLOAT_PRECISION = 4


def float_precision(precision=None):
    """
    Get or set the global float precision for timing outputs.

    Parameters
    ----------
    precision : int, optional
        Number of decimal places to display in timing outputs.
        If None, returns the current precision setting.

    Returns
    -------
    int
        Current precision value if precision=None, otherwise None.

    Examples
    --------
    To get the current precision, call ``float_precision()`` without an
    argument. To update it, call ``float_precision(6)``. All subsequent
    timing outputs then use six decimal places. Call ``float_precision(4)``
    to restore the default setting.
    """
    global _DEFAULT_FLOAT_PRECISION

    if precision is None:
        return _DEFAULT_FLOAT_PRECISION

    if not isinstance(precision, int) or precision < 0:
        raise ValueError("precision must be a non-negative integer")

    _DEFAULT_FLOAT_PRECISION = precision


def get_default_precision():
    """
    Get the current default precision setting.

    This is equivalent to calling :func:`float_precision` without an
    argument. Prefer :func:`float_precision` for new code because it also
    provides the corresponding setter.

    Returns
    -------
    int
        Current default precision for timing outputs.
    """
    return _DEFAULT_FLOAT_PRECISION
