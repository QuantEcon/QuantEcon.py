import numpy as np
from quantecon._interpolation import interp, _interpf, _interpa

import pytest
from numpy.testing import assert_allclose

def test_interp_float():
    xp = np.array([1, 2, 3])
    fp = np.array([3, 2, 0])
    numpy_y = np.interp(2.5, xp, fp)
    qe_y = _interpf(2.5, xp, fp)
    assert_allclose(qe_y, numpy_y)
    qe_y2 = interp(2.5, xp, fp)
    assert_allclose(qe_y2, numpy_y)


def test_interp_array():
    x = np.linspace(0, 2*np.pi, 10)
    y = np.sin(x)
    xvals = np.linspace(0, 2*np.pi, 50)
    numpy_y = np.interp(xvals, x, y)
    qe_y = _interpa(xvals, x, y)
    assert_allclose(qe_y, numpy_y)
    qe_y2 = interp(xvals, x, y)
    assert_allclose(qe_y2, numpy_y)