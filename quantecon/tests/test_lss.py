"""
Tests for lss.py

"""
import numpy as np
from numpy.testing import assert_allclose, assert_, assert_raises
from quantecon.lss import LinearStateSpace


class TestLinearStateSpace:

    def setup_method(self):
        # Example 1
        A = .95
        C = .05
        G = 1.
        mu_0 = .75

        self.ss1 = LinearStateSpace(A, C, G, mu_0=mu_0)

        # Example 2
        ρ1 = 0.5
        ρ2 = 0.3
        α = 0.5

        A = np.array([[ρ1, ρ2, α], [1, 0, 0], [0, 0, 1]])
        C = np.array([[1], [0], [0]])
        G = np.array([[1, 0, 0]])
        mu_0 = [0.5, 0.5, 1]

        self.ss2 = LinearStateSpace(A, C, G, mu_0=mu_0)

    def teardown_method(self):
        del self.ss1
        del self.ss2

    def test_stationarity(self):
        vals = self.ss1.stationary_distributions()
        ssmux, ssmuy, sssigx, sssigy, sssigyx = vals

        assert_(abs(ssmux - ssmuy) < 2e-8)
        assert_(abs(sssigx - sssigy) < 2e-8)
        assert_(abs(ssmux) < 2e-8)
        assert_(abs(sssigx - self.ss1.C**2/(1 - self.ss1.A**2)) < 2e-8)
        assert_(abs(sssigyx - self.ss1.G @ sssigx) < 2e-8)

        vals = self.ss2.stationary_distributions()
        ssmux, ssmuy, sssigx, sssigy, sssigyx = vals

        assert_allclose(ssmux.flatten(), np.array([2.5, 2.5, 1]))
        assert_allclose(ssmuy.flatten(), np.array([2.5]))
        assert_allclose(
            sssigx,
            self.ss2.A @ sssigx @ self.ss2.A.T + self.ss2.C @ self.ss2.C.T
        )
        assert_allclose(sssigy, self.ss2.G @ sssigx @ self.ss2.G.T)
        assert_allclose(sssigyx, self.ss2.G @ sssigx)

    def test_simulate(self):
        ss = self.ss1

        sim = ss.simulate(ts_length=250)
        for arr in sim:
            assert_(len(arr[0]) == 250)

    def test_simulate_with_seed(self):
        ss = self.ss1

        xval, yval = ss.simulate(ts_length=5, random_state=5)
        expected_output = np.array([0.75, 0.69595649, 0.78269723, 0.73095776,
                                    0.69989036])

        assert_allclose(xval[0], expected_output)
        assert_allclose(yval[0], expected_output)

    def test_replicate(self):
        xval, yval = self.ss1.replicate(T=100, num_reps=5000)

        assert_allclose(xval, yval)
        assert_(xval.size == 5000)
        assert_(abs(np.mean(xval)) <= .05)

    def test_replicate_with_seed(self):
        xval, yval = self.ss1.replicate(T=100, num_reps=5, random_state=5)
        expected_output = np.array([0.10498898, 0.02892168, 0.04915998,
                                    0.18568489, 0.04541764])

        assert_allclose(xval[0], expected_output)
        assert_allclose(yval[0], expected_output)


def test_non_square_A():
    A = np.zeros((1, 2))
    C = np.zeros((1, 1))
    G = np.zeros((1, 1))

    assert_raises(ValueError, LinearStateSpace, A, C, G)
