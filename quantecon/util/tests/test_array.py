"""
Tests for Array Utilities

Functions
---------
searchsorted

"""
import numpy as np
from numpy.testing import assert_
from numba import njit
import pytest
from quantecon.util import searchsorted


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_searchsorted():
    a = np.array([0.2, 0.4, 1.0])
    assert_(searchsorted(a, 0.1) == 0)
    assert_(searchsorted(a, 0.4) == 2)
    assert_(searchsorted(a, 2) == 3)

    a = np.ones(0)
    for (v, i) in zip([0, 1, 2], [0, 0, 0]):
        assert_(searchsorted(a, v) == i)

    a = np.ones(1)
    for (v, i) in zip([0, 1, 2], [0, 1, 1]):
        assert_(searchsorted(a, v) == i)

    a = np.ones(2)
    for (v, i) in zip([0, 1, 2], [0, 2, 2]):
        assert_(searchsorted(a, v) == i)


@njit
def _jitted_function():
    a = np.array([0.2, 0.4, 1.0])
    return searchsorted(a, 0.5)


def test_warns():
    a = np.array([0.2, 0.4, 1.0])
    with pytest.warns(DeprecationWarning):
        searchsorted(a, 0.5)

    with pytest.warns(DeprecationWarning):
        _jitted_function()
