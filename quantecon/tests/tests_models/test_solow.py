"""
Test suite for solow module.

@author : David R. Pugh
@date : 2014-09-01

"""
import nose

import numpy as np
import sympy as sym

from ... models import solow

# declare key variables for the model
t, X = sym.var('t'), sym.DeferredVector('X')
A, E, k, K, L = sym.var('A, E, k, K, L')

# declare required model parameters
g, n, s, alpha, delta = sym.var('g, n, s, alpha, delta')


# use the Solow Model with Cobb-Douglas production as test case
def invalid_output(A, K, L, alpha):
    """Output must be of type sym.basic, not function."""
    return K**alpha * (A * L)**(1 - alpha)


def _k_upper(g, n, s, alpha, delta):
    """Upper bound on possible steady state value."""
    return (1 / (g + n + delta))**(1 / (1 - alpha))


valid_output = K**alpha * (A * L)**(1 - alpha)

invalid_params = {'g': -0.02, 'n': -0.02, 's': 0.15, 'alpha': 0.33, 'delta': 0.03}
valid_params = {'g': 0.02, 'n': 0.02, 's': 0.15, 'alpha': 0.33, 'delta': 0.05}


# testing functions
def test_validate_output():
    """Testing validation of output attribute."""
    with nose.tools.assert_raises(AttributeError):
        solow.Model(output=invalid_output, params=valid_params)

    mod = solow.Model(output=valid_output, params=valid_params)
    nose.tools.assert_equals(valid_output, mod.output)


def test_validate_params():
    """Testing validation of output attribute."""
    with nose.tools.assert_raises(AttributeError):
        solow.Model(output=valid_output, params=invalid_params)

    mod = solow.Model(output=valid_output, params=valid_params)
    nose.tools.assert_equals(valid_params, mod.params)


def test_find_steady_state():
    """Testing computation of steady state."""
    for g in np.linspace(-0.02, 0.02, 4):
        for n in np.linspace(-0.02, 0.03, 4):
            for s in np.linspace(1e-2, 1-1e-2, 4):
                for alpha in np.linspace(1e-2, 1-1e-2, 4):
                    for delta in np.linspace(0.05, 1-1e-2, 4):

                        tmp_params = {'g': g, 'n': n, 's': s, 'alpha': alpha, 'delta': delta}
                        tmp_mod = solow.Model(output=valid_output,
                                                    params=tmp_params)
                        actual_steady_state = tmp_mod.find_steady_state(0, _k_upper(**tmp_params))
                        expected_steady_state = solow.cobb_douglas.analytic_steady_state(**tmp_params)

                        nose.tools.assert_almost_equals(actual_steady_state, expected_steady_state)
