"""
Test suite for solow.cobb_douglas.py module.

@author : David R. Pugh

"""
import nose

import numpy as np

from .... models.solow import cobb_douglas

params = {'A0': 1.0, 'g': 0.02, 'L0': 1.0, 'n': 0.02, 's': 0.15,
          'alpha': 0.33, 'delta': 0.05}
model = cobb_douglas.CobbDouglasModel(params)


def test_ivp_solve():
    """Testing computation of solution to the initial value problem."""
    eps = 1e-1
    for g in np.linspace(eps, 0.05, 4):
        for n in np.linspace(eps, 0.05, 4):
            for s in np.linspace(eps, 1-eps, 4):
                for alpha in np.linspace(eps, 1-eps, 4):
                    for delta in np.linspace(eps, 1-eps, 4):

                        tmp_params = {'A0': 1.0, 'g': g, 'L0': 1.0, 'n': n,
                                      's': s, 'alpha': alpha, 'delta': delta}
                        model.params = tmp_params

                        # solve the initial value problem
                        t0, k0 = 0, 0.5 * model.steady_state
                        numeric_soln = model.ivp.solve(t0, k0, T=100)

                        # compute the analytic solution
                        tmp_ti = numeric_soln[:, 0]
                        analytic_soln = model.analytic_solution(tmp_ti, k0)

                        # conduct the test
                        np.testing.assert_allclose(numeric_soln, analytic_soln)


def test_root_finders():
    """Testing conditional logic in find_steady_state."""
    valid_methods = ['brenth', 'brentq', 'ridder', 'bisect']
    for method in valid_methods:
        actual_ss = model.find_steady_state(1e-6, 1e6, method=method)
        expected_ss = model.steady_state
        nose.tools.assert_almost_equals(actual_ss, expected_ss)


def test_steady_state():
    """Compare analytic steady state with numerical steady state."""
    eps = 1e-1
    for g in np.linspace(eps, 0.05, 4):
        for n in np.linspace(eps, 0.05, 4):
            for s in np.linspace(eps, 1-eps, 4):
                for alpha in np.linspace(eps, 1-eps, 4):
                    for delta in np.linspace(eps, 1-eps, 4):

                        tmp_params = {'A0': 1.0, 'g': g, 'L0': 1.0, 'n': n,
                                      's': s, 'alpha': alpha, 'delta': delta}
                        model.params = tmp_params

                        # use root finder to compute the steady state
                        actual_ss = model.steady_state
                        expected_ss = model.find_steady_state(1e-12, 1e12)

                        # conduct the test
                        nose.tools.assert_almost_equals(actual_ss, expected_ss)


def test_valid_methods():
    """Testing invalid method passed to find_steady_state."""
    with nose.tools.assert_raises(ValueError):
        model.find_steady_state(1e-12, 1e12, method='invalid_method')


def test_valid_parameters():
    """Testing invalid value for output elasticity."""
    with nose.tools.assert_raises(AttributeError):
        invalid_params = {'A0': 1.0, 'g': 0.02, 'L0': 1.0, 'n': 0.02,
                          's': 0.15, 'alpha': 1.1, 'delta': 0.03}
        cobb_douglas.CobbDouglasModel(invalid_params)
