"""
Test suite for solow module.

@author : David R. Pugh
@date : 2014-09-01

"""
import nose

import sympy as sym

from ... models import solow

# declare key variables for the model
t, X = sym.var('t'), sym.DeferredVector('X')
A, E, k, K, L = sym.var('A, E, k, K, L')

# declare required model parameters
g, n, s, alpha, delta = sym.var('g, n, s, alpha, delta')

# use the Solow Model with Cobb-Douglas production as test case
valid_output = K**alpha * (A * L)**(1 - alpha)


def invalid_output(A, K, L, alpha):
    """Output must be of type sym.basic, not function."""
    return K**alpha * (A * L)**(1 - alpha)

valid_params = {'g': 0.02, 'n': 0.02, 's': 0.15, 'alpha': 0.33, 'delta': 0.05}
invalid_params = {'g': -0.02, 'n': -0.02, 's': 0.15, 'alpha': 0.33, 'delta': 0.03}


# testing functions
def test_validate_output():
    """Testing validation of output attribute."""
    with nose.tools.assert_raises(AttributeError):
        model = solow.model.Model(output=invalid_output, params=valid_params)
