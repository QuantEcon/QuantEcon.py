"""
Test suite for solow module.

@author : David R. Pugh
@date : 2014-09-01

"""
import nose

#import matplotlib
#matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import sympy as sym

from ... models import solow
from ... models.solow import cobb_douglas

# declare key variables for the model
t, X = sym.var('t'), sym.DeferredVector('X')
A, E, k, K, L = sym.var('A, E, k, K, L')

# declare required model parameters
g, n, s, alpha, delta = sym.var('g, n, s, alpha, delta')


# two different ways in which output can fail
def invalid_output_1(A, K, L, alpha):
    """Output must be of type sym.basic, not function."""
    return K**alpha * (A * L)**(1 - alpha)

invalid_output_2 = K**alpha * (A * E)**(1 - alpha)

valid_output = K**alpha * (A * L)**(1 - alpha)

valid_params = {'A0': 1.0, 'g': 0.02, 'L0': 1.0, 'n': 0.02, 's': 0.15,
                'alpha': 0.33, 'delta': 0.05}


# testing functions
def test_validate_output():
    """Testing validation of output attribute."""
    # output must have type sym.Basic
    with nose.tools.assert_raises(AttributeError):
        solow.Model(output=invalid_output_1, params=valid_params)

    # output must be function of K, A, L
    with nose.tools.assert_raises(AttributeError):
        solow.Model(output=invalid_output_2, params=valid_params)


def test_validate_params():
    """Testing validation of params attribute."""
    # four different ways in which params can fail
    invalid_params_0 = (1.0, 1.0, 0.02, 0.02, 0.15, 0.33, 0.03)
    invalid_params_1 = {'A0': 1.0, 'g': -0.02, 'L0': 1.0, 'n': -0.02, 's': 0.15,
                        'alpha': 0.33, 'delta': 0.03}
    invalid_params_2 = {'A0': 1.0, 'g': 0.02, 'L0': 1.0, 'n': 0.02, 's': 0.15,
                        'alpha': 0.33, 'delta': -0.03}
    invalid_params_3 = {'A0': 1.0, 'g': 0.02, 'L0': 1.0, 'n': 0.02, 's': -0.15,
                        'alpha': 0.33, 'delta': 0.03}

    # params must be a dict
    with nose.tools.assert_raises(AttributeError):
        solow.Model(output=valid_output, params=invalid_params_0)

    with nose.tools.assert_raises(AttributeError):
        solow.CobbDouglasModel(params=invalid_params_0)

    # effective depreciation rate must be positive
    with nose.tools.assert_raises(AttributeError):
        solow.Model(output=valid_output, params=invalid_params_1)

    with nose.tools.assert_raises(AttributeError):
        solow.CobbDouglasModel(params=invalid_params_1)

    # physical depreciation rate must be positive
    with nose.tools.assert_raises(AttributeError):
        solow.Model(output=valid_output, params=invalid_params_2)

    with nose.tools.assert_raises(AttributeError):
        solow.CobbDouglasModel(params=invalid_params_2)

    # savings rate must be positive
    with nose.tools.assert_raises(AttributeError):
        solow.Model(output=valid_output, params=invalid_params_3)

    with nose.tools.assert_raises(AttributeError):
        solow.CobbDouglasModel(params=invalid_params_3)


def test_find_steady_state():
    """Testing computation of steady state."""
    eps = 1e-1
    for g in np.linspace(eps, 0.05, 4):
        for n in np.linspace(eps, 0.05, 4):
            for s in np.linspace(eps, 1-eps, 4):
                for alpha in np.linspace(eps, 1-eps, 4):
                    for delta in np.linspace(eps, 1-eps, 4):

                        tmp_params = {'A0': 1.0, 'g': g, 'L0': 1.0, 'n': n,
                                      's': s, 'alpha': alpha, 'delta': delta}
                        tmp_mod = solow.Model(output=valid_output,
                                              params=tmp_params)

                        # use root finder to compute the steady state
                        actual_ss = tmp_mod.steady_state
                        expected_ss = cobb_douglas.analytic_steady_state(tmp_mod)

                        # conduct the test (default places=7 is too precise!)
                        nose.tools.assert_almost_equals(actual_ss, expected_ss,
                                                        places=6)


def test_compute_output_elsticity():
    """Testing computation of elasticity of output with respect to capital."""
    eps = 1e-1
    for g in np.linspace(eps, 0.05, 4):
        for n in np.linspace(eps, 0.05, 4):
            for s in np.linspace(eps, 1-eps, 4):
                for alpha in np.linspace(eps, 1-eps, 4):
                    for delta in np.linspace(eps, 1-eps, 4):

                        tmp_params = {'A0': 1.0, 'g': g, 'L0': 1.0, 'n': n,
                                      's': s, 'alpha': alpha, 'delta': delta}
                        tmp_mod = solow.Model(output=valid_output,
                                              params=tmp_params)

                        # use root finder to compute the steady state
                        tmp_k_star = tmp_mod.steady_state

                        actual_elasticity = tmp_mod.compute_output_elasticity(tmp_k_star)
                        expected_elasticity = tmp_params['alpha']

                        # conduct the test
                        nose.tools.assert_almost_equals(actual_elasticity,
                                                        expected_elasticity)


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
                        tmp_mod = solow.Model(output=valid_output,
                                              params=tmp_params)

                        # use root finder to compute the steady state
                        tmp_k_star = tmp_mod.steady_state

                        # solve the initial value problem
                        t0, k0 = 0, 0.5 * tmp_k_star
                        numeric_soln = tmp_mod.ivp.solve(t0, k0, T=100)

                        tmp_ti = numeric_soln[:, 0]
                        analytic_soln = cobb_douglas.analytic_solution(tmp_mod, tmp_ti, k0)

                        # conduct the test (default places=7 is too precise!)
                        np.testing.assert_allclose(numeric_soln, analytic_soln)


def test_root_finders():
    """Testing conditional logic in find_steady_state."""
    # loop over solvers
    tmp_mod = solow.Model(output=valid_output, params=valid_params)
    valid_methods = ['brenth', 'brentq', 'ridder', 'bisect']

    for method in valid_methods:
        actual_ss = tmp_mod.find_steady_state(1e-6, 1e6, method=method)
        expected_ss = cobb_douglas.analytic_steady_state(tmp_mod)
        nose.tools.assert_almost_equals(actual_ss, expected_ss)


def test_valid_methods():
    """Testing raise exception if invalid method passed to find_steady_state."""
    with nose.tools.assert_raises(ValueError):
        tmp_mod = solow.Model(output=valid_output, params=valid_params)
        tmp_mod.find_steady_state(1e-12, 1e12, method='invalid_method')
