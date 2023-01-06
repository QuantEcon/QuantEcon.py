"""
Tests for markov/estimate.py

"""
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from quantecon import estimate_mc
from quantecon.markov import fit_discrete_mc


class TestEstimateMCDiscrete:
    def setup_method(self):
        self.test_index_series = np.array((0, 1, 1, 1, 1, 0, 2, 1))
        self.initial_state_values = np.array(['a', 'b', 'c', 'd'])
        self.test_value_series = \
            self.initial_state_values[self.test_index_series]

        self.P = np.array([[0.,   0.5,  0.5 ],
                           [0.25, 0.75, 0.  ],
                           [0.,   1.,   0.  ]])

        self.final_state_indices = np.array([0, 1, 2])
        self.final_state_values = \
            self.initial_state_values[self.final_state_indices]

    def test_integer_state(self):
        mc = estimate_mc(self.test_index_series)
        assert_allclose(mc.P, self.P)
        assert_array_equal(mc.state_values, self.final_state_indices)

    def test_non_integer_state(self):
        mc = estimate_mc(self.test_value_series)
        assert_allclose(mc.P, self.P)
        assert_array_equal(mc.state_values, self.final_state_values)

        mc = estimate_mc(self.test_index_series)
        mc.state_values = self.initial_state_values[mc.state_values]
        assert_allclose(mc.P, self.P)
        assert_array_equal(mc.state_values, self.final_state_values)

    def test_mult_dim_state(self):
        initial_state_values = np.array([[0.97097089, 0.76167618],
                                         [0.61878456, 0.41691572],
                                         [0.42137226, 0.09307409],
                                         [0.62696609, 0.40898893]])
        X = initial_state_values[self.test_index_series]
        ind = np.lexsort(
            np.rot90(initial_state_values[self.final_state_indices])
        )
        final_state_values = initial_state_values[ind]

        mc = estimate_mc(X)
        assert_allclose(mc.P, self.P[np.ix_(ind, ind)])
        assert_array_equal(mc.state_values, final_state_values)


class TestFitDiscreteMC:
    def setup_method(self):
        self.grids = ((np.arange(4), np.arange(5)))
        self.X = [(-0.1, 1.2), (2, 0), (2, 3),
                  (4.4, 4.0), (0.6, 0.4), (1.0, 0.1)]

    def test_order_f(self):
        mc = fit_discrete_mc(self.X, self.grids, order='F')
        S_expected = np.array([[1, 0], [2, 0], [0, 1], [2, 3], [3, 4]])
        P_expected = np.array([[1., 0., 0., 0., 0.],
                               [0., 0., 0., 1., 0.],
                               [0., 1., 0., 0., 0.],
                               [0., 0., 0., 0., 1.],
                               [1., 0., 0., 0., 0.]])

        assert_array_equal(mc.state_values, S_expected)
        assert_allclose(mc.P, P_expected)

    def test_order_c(self):
        mc = fit_discrete_mc(self.X, self.grids, order='C')
        S_expected = np.array([[0, 1], [1, 0], [2, 0], [2, 3], [3, 4]])
        P_expected = np.array([[0., 0., 1., 0., 0.],
                               [0., 1., 0., 0., 0.],
                               [0., 0., 0., 1., 0.],
                               [0., 0., 0., 0., 1.],
                               [0., 1., 0., 0., 0.]])

        assert_array_equal(mc.state_values, S_expected)
        assert_allclose(mc.P, P_expected)
