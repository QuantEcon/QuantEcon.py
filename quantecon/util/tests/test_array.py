"""
Tests for Array Utilities

Functions
---------
searchsorted

"""
import numpy as np
from numpy.testing import assert_
from quantecon.util import searchsorted


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
