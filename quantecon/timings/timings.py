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
    Get current precision:
    >>> import quantecon as qe
    >>> current = qe.timings.float_precision()
    >>> print(f"Current precision: {current}")
    
    Set new precision:
    >>> qe.timings.float_precision(6)
    >>> # All subsequent timing outputs will use 6 decimal places
    
    Reset to default:
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
    
    Returns
    -------
    int
        Current default precision for timing outputs.
    """
    return _DEFAULT_FLOAT_PRECISION