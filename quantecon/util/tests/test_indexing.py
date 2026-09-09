"""
Tests for Indexing Utilities

Functions
---------
index_dict

"""
import numpy as np
from numpy.testing import assert_
from numba import njit
import pytest
from quantecon.util import index_dict


class TestIndexDict1D:
    def setup_method(self):
        self.vals = np.array([0.5, -1.0, 2.5])
        self.d = index_dict(self.vals)

    def test_lookup(self):
        for i, v in enumerate(self.vals):
            assert_(self.d[v] == i)

    def test_length(self):
        assert_(len(self.d) == len(self.vals))

    def test_use_in_jitted_function(self):
        @njit
        def f(dd, v):
            return dd[v]

        assert_(f(self.d, -1.0) == 1)

    def test_pass_as_argument_and_modify(self):
        @njit
        def contains(dd, v):
            return v in dd

        assert_(contains(self.d, 2.5))
        assert_(not contains(self.d, 10.0))


class TestIndexDict2D:
    def setup_method(self):
        self.vals = np.array([[0.0, 0.1], [0.0, 1.0], [0.5, 0.1],
                              [0.5, 1.0]])
        self.d = index_dict(self.vals)

    def test_lookup(self):
        for i in range(self.vals.shape[0]):
            assert_(self.d[tuple(self.vals[i])] == i)

    def test_use_in_jitted_function(self):
        @njit
        def f(dd, row):
            return dd[(row[0], row[1])]

        assert_(f(self.d, self.vals[2]) == 2)


def test_index_dict_3_columns():
    vals = np.arange(12.).reshape(4, 3)
    d = index_dict(vals)
    for i in range(vals.shape[0]):
        assert_(d[tuple(vals[i])] == i)


def test_index_dict_int_dtype():
    vals = np.array([2, 0, 5])
    d = index_dict(vals)
    for i, v in enumerate(vals):
        assert_(d[v] == i)


def test_index_dict_int_dtype_2d():
    vals = np.array([[0, 1], [1, 0]])
    d = index_dict(vals)
    assert_(d[(1, 0)] == 1)


def test_index_dict_array_like():
    d = index_dict([0.1, 0.2])
    assert_(d[0.2] == 1)


def test_index_dict_empty():
    d = index_dict(np.empty(0))
    assert_(len(d) == 0)


def test_index_dict_duplicates_raise():
    with pytest.raises(ValueError, match="duplicate value"):
        index_dict(np.array([0.5, 1.0, 0.5]))


def test_index_dict_duplicates_raise_2d():
    with pytest.raises(ValueError, match="duplicate value"):
        index_dict(np.array([[0.5, 1.0], [0.5, 1.0]]))


def test_index_dict_negative_zero_raises():
    # Numba typed dicts hash floats by bit pattern, so -0.0 and 0.0
    # would be distinct keys; -0.0 is rejected at build time
    with pytest.raises(ValueError, match="-0.0"):
        index_dict(np.array([1.0, -0.0]))
    with pytest.raises(ValueError, match="-0.0"):
        index_dict(np.array([[1.0, -0.0]]))


def test_index_dict_positive_zero_ok():
    d = index_dict(np.array([0.0, 1.0]))
    assert_(d[0.0] == 0)


def test_index_dict_nan_raises():
    with pytest.raises(ValueError, match="non-finite"):
        index_dict(np.array([0.5, np.nan]))


def test_index_dict_inf_raises():
    with pytest.raises(ValueError, match="non-finite"):
        index_dict(np.array([[0.5, np.inf]]))


def test_index_dict_invalid_ndim():
    with pytest.raises(ValueError, match="1- or 2-dimensional"):
        index_dict(np.zeros((2, 2, 2)))
    with pytest.raises(ValueError, match="1- or 2-dimensional"):
        index_dict(np.float64(1.0))


def test_index_dict_zero_columns():
    with pytest.raises(ValueError, match="at least one column"):
        index_dict(np.empty((3, 0)))


def test_index_dict_invalid_dtype():
    with pytest.raises(ValueError, match="unsupported dtype"):
        index_dict(np.array(['a', 'b']))
