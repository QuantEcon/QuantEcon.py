"""
Tests for markov/estimate.py

"""
import numpy as np

from numpy.testing import assert_array_equal
from numpy.testing import assert_allclose


def test_est_integer_state():
    test_series = (0, 1, 1, 1, 1, 0, 2, 1)
    input_state_values = (0, 1, 2, 3)
    P = [[0.,   0.5,  0.5 ],
         [0.25, 0.75, 0.  ],
         [0.,   1.,   0.  ]]
    P = np.asarray(P)
    final_state_values = np.array([0, 1, 2])

    mc = estimate_mc_discrete(test_series, input_state_values)
    assert_allclose(mc.P, P)
    assert_array_equal(mc.state_values, final_state_values)


def test_est_non_integer_state():
    test_series = (0, 1, 1, 1, 1, 0, 2, 0)
    input_state_values = ('a', 'b', 'c', 'd')
    P = [[0.,   0.5,  0.5 ],
         [0.25, 0.75, 0.  ],
         [1.,   0.,   0.  ]]
    P = np.asarray(P)
    final_state_values = np.array(('a', 'b', 'c'))

    mc = estimate_mc_discrete(test_series, input_state_values)
    assert_allclose(mc.P, P)
    assert_array_equal(mc.state_values == final_state_values)
