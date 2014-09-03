"""
Test suite for solow module.

@author : David R. Pugh
@date : 2014-09-01

"""
import nose

import numpy as np
import sympy as sym

from ... models import solow
from ... models.solow.cobb_douglas import analytic_steady_state

# declare key variables for the model
t, X = sym.var('t'), sym.DeferredVector('X')
A, E, k, K, L = sym.var('A, E, k, K, L')

# declare required model parameters
g, n, s, alpha, delta = sym.var('g, n, s, alpha, delta')


# use the Solow Model with Cobb-Douglas production as test case
def invalid_output_1(A, K, L, alpha):
    """Output must be of type sym.basic, not function."""
    return K**alpha * (A * L)**(1 - alpha)

invalid_output_2 = K**alpha * (A * E)**(1 - alpha)
valid_output = K**alpha * (A * L)**(1 - alpha)

# four different ways in which params can fail
invalid_params_0 = (0.02, 0.02, 0.15, 0.33, 0.03)
invalid_params_1 = {'g': -0.02, 'n': -0.02, 's': 0.15, 'alpha': 0.33, 'delta': 0.03}
invalid_params_2 = {'g': 0.02, 'n': 0.02, 's': 0.15, 'alpha': 0.33, 'delta': -0.03}
invalid_params_3 = {'g': 0.02, 'n': 0.02, 's': -0.15, 'alpha': 0.33, 'delta': 0.03}

valid_params = {'g': 0.02, 'n': 0.02, 's': 0.15, 'alpha': 0.33, 'delta': 0.05}


# helper functions
def k_upper(g, n, s, alpha, delta):
    """Upper bound on possible steady state value."""
    return (1 / (g + n + delta))**(1 / (1 - alpha))


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
    """Testing validation of output attribute."""
    # params must be a dict
    with nose.tools.assert_raises(AttributeError):
        solow.Model(output=valid_output, params=invalid_params_0)

    # effective depreciation rate must be positive
    with nose.tools.assert_raises(AttributeError):
        solow.Model(output=valid_output, params=invalid_params_1)

    # physical depreciation rate must be positive
    with nose.tools.assert_raises(AttributeError):
        solow.Model(output=valid_output, params=invalid_params_2)

    # savings rate must be positive
    with nose.tools.assert_raises(AttributeError):
        solow.Model(output=valid_output, params=invalid_params_3)


def test_find_steady_state():
    """Testing computation of steady state."""
    # loop over different parameters values
    eps = 1e-1
    for g in np.linspace(eps, 0.05, 4):
        for n in np.linspace(eps, 0.05, 4):
            for s in np.linspace(eps, 1-eps, 4):
                for alpha in np.linspace(eps, 1-eps, 4):
                    for delta in np.linspace(eps, 1-eps, 4):

                        tmp_params = {'g': g, 'n': n, 's': s, 'alpha': alpha,
                                      'delta': delta}
                        tmp_mod = solow.Model(output=valid_output,
                                              params=tmp_params)

                        # use root finder to compute the steady state
                        tmp_k_upper = k_upper(**tmp_params)
                        actual_ss = tmp_mod.find_steady_state(1e-12, tmp_k_upper)
                        expected_ss = analytic_steady_state(**tmp_params)

                        # conduct the test (default places=7 is too precise!)
                        nose.tools.assert_almost_equals(actual_ss, expected_ss,
                                                        places=6)


def test_root_finders():
    """Testing conditional logic in find_steady_state."""
    # loop over solvers
    tmp_mod = solow.Model(output=valid_output, params=valid_params)
    valid_methods = ['brenth', 'brentq', 'ridder', 'bisect']

    for method in valid_methods:
        tmp_k_upper = k_upper(**valid_params)
        actual_ss = tmp_mod.find_steady_state(1e-12, tmp_k_upper, method=method)
        expected_ss = analytic_steady_state(**valid_params)
        nose.tools.assert_almost_equals(actual_ss, expected_ss)


def test_valid_methods():
    """Testing raise exception if invalid method passed to find_steady_state."""
    with nose.tools.assert_raises(ValueError):
        tmp_mod = solow.Model(output=valid_output, params=valid_params)
        tmp_mod.find_steady_state(1e-12, k_upper(**valid_params),
                                  method='invalid_method')
