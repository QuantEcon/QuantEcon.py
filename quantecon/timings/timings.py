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
    Get the current precision:

    >>> import quantecon as qe
    >>> qe.timings.float_precision()
    4

    Set a new precision. All subsequent timing outputs use six decimal
    places:

    >>> qe.timings.float_precision(6)

    Reset to the default:

    >>> qe.timings.float_precision(4)
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

    This is a read-only equivalent of calling :func:`float_precision` with
    no argument; both return the same module-level value. It is the
    accessor used internally by the timing utilities in
    :mod:`quantecon.util.timing`, which only need to read the setting.

    Returns
    -------
    int
        Current default precision for timing outputs.
    """
    return _DEFAULT_FLOAT_PRECISION
