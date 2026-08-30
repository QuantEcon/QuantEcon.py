"""Tests for random state utilities."""

import ast
import inspect
from pathlib import Path

import numpy as np
from numpy.testing import assert_array_equal

from quantecon.util import random as random_utils
from quantecon.util import rng_integers


def _expected_random_state_call(random_state, low, high, size, dtype, endpoint):
    if high is None:
        high = low + 1 if endpoint else low
        return random_state.randint(high, size=size, dtype=dtype)
    if endpoint:
        high += 1
    return random_state.randint(low, high=high, size=size, dtype=dtype)


def test_rng_integers_has_no_private_scipy_import():
    source = Path(random_utils.__file__).read_text()
    tree = ast.parse(source)
    private_module = '.'.join(('scipy', '_lib', '_util'))
    private_imports = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == private_module
    ]
    assert not private_imports


def test_rng_integers_signature_is_stable():
    assert str(inspect.signature(rng_integers)) == (
        "(gen, low, high=None, size=None, dtype='int64', endpoint=False)"
    )


def test_rng_integers_matches_random_state():
    cases = [
        (0, 10, (32,), np.int64, False),
        (-5, 5, (32,), np.int32, True),
        (4, None, 32, np.int16, True),
    ]

    for low, high, size, dtype, endpoint in cases:
        actual = rng_integers(
            np.random.RandomState(12345),
            low,
            high=high,
            size=size,
            dtype=dtype,
            endpoint=endpoint,
        )
        expected = _expected_random_state_call(
            np.random.RandomState(12345),
            low,
            high,
            size,
            dtype,
            endpoint,
        )
        assert_array_equal(actual, expected)
        assert actual.dtype == np.dtype(dtype)


def test_rng_integers_matches_generator():
    cases = [
        (0, 10, (32,), np.int64, False),
        (-5, 5, (32,), np.int32, True),
        (4, None, 32, np.int16, True),
    ]

    for low, high, size, dtype, endpoint in cases:
        actual = rng_integers(
            np.random.default_rng(12345),
            low,
            high=high,
            size=size,
            dtype=dtype,
            endpoint=endpoint,
        )
        expected = np.random.default_rng(12345).integers(
            low,
            high=high,
            size=size,
            dtype=dtype,
            endpoint=endpoint,
        )
        assert_array_equal(actual, expected)
        assert actual.dtype == np.dtype(dtype)


def test_rng_integers_none_uses_random_state_singleton():
    original_state = np.random.get_state()
    try:
        np.random.seed(12345)
        actual = rng_integers(None, -3, high=4, size=32, endpoint=True)
        np.random.seed(12345)
        expected = np.random.randint(-3, high=5, size=32)
    finally:
        np.random.set_state(original_state)

    assert_array_equal(actual, expected)
