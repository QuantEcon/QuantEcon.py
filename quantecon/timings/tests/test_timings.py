"""
Tests for quantecon/timings/timings.py

Covers the previously zero-coverage public API:

* ``float_precision`` getter (precision=None) and setter
* ``ValueError`` paths for a negative int, a float, and a string
* ``get_default_precision`` returning the current global value
* global-state restoration so tests never leak precision changes
"""
import pytest

import quantecon as qe


@pytest.fixture(autouse=True)
def restore_precision():
    """Snapshot the global precision before each test and restore it after."""
    original = qe.timings.get_default_precision()
    yield original
    qe.timings.float_precision(original)


def test_get_default_precision_returns_current(restore_precision):
    # get_default_precision reports the live global value, not a constant.
    assert qe.timings.get_default_precision() == restore_precision


def test_float_precision_getter_returns_current(restore_precision):
    # Calling with no argument returns the current precision.
    assert qe.timings.float_precision() == restore_precision


def test_float_precision_setter_changes_value():
    # Setting must be observable via both the getter and get_default_precision.
    qe.timings.float_precision(6)
    assert qe.timings.float_precision() == 6
    assert qe.timings.get_default_precision() == 6


def test_float_precision_setter_accepts_zero():
    # Zero is a valid non-negative integer.
    qe.timings.float_precision(0)
    assert qe.timings.float_precision() == 0


def test_float_precision_setter_accepts_large_value():
    qe.timings.float_precision(12)
    assert qe.timings.float_precision() == 12


@pytest.mark.parametrize("bad", [-1, 4.5, "4"])
def test_float_precision_value_error_on_invalid(bad):
    # Negative int, float, and string must all raise ValueError.
    with pytest.raises(ValueError, match="non-negative integer"):
        qe.timings.float_precision(bad)


def test_float_precision_value_error_does_not_change_state():
    # A failed set must leave the global precision untouched.
    before = qe.timings.get_default_precision()
    with pytest.raises(ValueError):
        qe.timings.float_precision(-1)
    assert qe.timings.get_default_precision() == before
