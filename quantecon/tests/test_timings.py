"""Tests for the public timing precision configuration API."""

import pytest

from quantecon.timings import float_precision, get_default_precision


@pytest.fixture(autouse=True)
def restore_float_precision():
    """Keep the module-level timing precision isolated between tests."""
    original_precision = float_precision()
    yield
    float_precision(original_precision)


def test_float_precision_gets_and_sets_the_default():
    float_precision(6)

    assert float_precision() == 6
    assert get_default_precision() == 6


@pytest.mark.parametrize("precision", [-1, 1.5, "4"])
def test_float_precision_rejects_invalid_values(precision):
    with pytest.raises(ValueError, match="non-negative integer"):
        float_precision(precision)


def test_float_precision_examples_are_not_uncollected_doctests():
    assert ">>>" not in float_precision.__doc__
