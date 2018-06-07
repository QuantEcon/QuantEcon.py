"""
Tests for scalar maximization.

"""
import numpy as np
from numpy.testing import assert_almost_equal
from numba import njit

from quantecon.optimize import maximize_scalar

@njit
def f(x):
    """
    A function for testing on.
    """
    return -(x + 2.0)**2 + 1.0

def test_maximize_scalar():
    """
    Uses the function f defined above to test the scalar maximization 
    routine.
    """
    true_fval = 1.0
    true_xf = -2.0
    fval, xf = maximize_scalar(f, -2, 2)
    assert_almost_equal(true_fval, fval, decimal=4)
    assert_almost_equal(true_xf, xf, decimal=4)
    
@njit
def g(x, y):
    """
    A multivariate function for testing on.
    """
    return -x**2 + y
    
def test_maximize_scalar_multivariate():
    """
    Uses the function f defined above to test the scalar maximization 
    routine.
    """
    y = 5
    true_fval = 5.0
    true_xf = -0.0
    fval, xf = maximize_scalar(g, -10, 10, args=(y,))
    assert_almost_equal(true_fval, fval, decimal=4)
    assert_almost_equal(true_xf, xf, decimal=4)


if __name__ == '__main__':
    import sys
    import nose

    argv = sys.argv[:]
    argv.append('--verbose')
    argv.append('--nocapture')
    nose.main(argv=argv, defaultTest=__file__)


