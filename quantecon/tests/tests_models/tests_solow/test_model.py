"""
Test suite for solow module.

@author : David R. Pugh
@date : 2014-11-27

"""
from __future__ import division
import nose

import matplotlib.pyplot as plt
import numpy as np
import sympy as sym

from .... models import solow

# declare key variables for the model
A, E, k, K, L = sym.symbols('A, E, k, K, L')

# declare required model parameters
g, n, s, alpha, delta = sym.symbols('g, n, s, alpha, delta')


# two different ways in which output can fail
def invalid_output_1(A, K, L, alpha):
    """Output must be of type sym.basic, not function."""
    return K**alpha * (A * L)**(1 - alpha)

invalid_output_2 = K**alpha * (A * E)**(1 - alpha)

valid_output = K**alpha * (A * L)**(1 - alpha)

valid_params = {'A0': 1.0, 'g': 0.02, 'L0': 1.0, 'n': 0.02, 's': 0.15,
                'alpha': 0.33, 'delta': 0.05}


# testing functions
def test_plot_factor_shares():
    """Testing return type for plot_factor_shares."""
    tmp_mod = solow.Model(output=valid_output, params=valid_params)
    fig, ax = plt.subplots(1, 1)
    tmp_lines = tmp_mod.plot_factor_shares(ax)
    nose.tools.assert_is_instance(tmp_lines, list)


def test_plot_intensive_investment():
    """Testing return type for plot_intensive_investment."""
    tmp_mod = solow.Model(output=valid_output, params=valid_params)
    fig, ax = plt.subplots(1, 1)
    tmp_lines = tmp_mod.plot_intensive_investment(ax)
    nose.tools.assert_is_instance(tmp_lines, list)


def test_plot_intensive_output():
    """Testing return type for plot_intensive_output."""
    tmp_mod = solow.Model(output=valid_output, params=valid_params)
    fig, ax = plt.subplots(1, 1)
    tmp_lines = tmp_mod.plot_intensive_output(ax)
    nose.tools.assert_is_instance(tmp_lines, list)


def test_plot_phase_diagram():
    """Testing return type for plot_phase_diagram."""
    tmp_mod = solow.Model(output=valid_output, params=valid_params)
    fig, ax = plt.subplots(1, 1)
    tmp_lines = tmp_mod.plot_phase_diagram(ax)
    nose.tools.assert_is_instance(tmp_lines, list)


def test_plot_solow_diagram():
    """Testing return type for plot_solow_diagram."""
    tmp_mod = solow.Model(output=valid_output, params=valid_params)
    fig, ax = plt.subplots(1, 1)
    tmp_lines = tmp_mod.plot_solow_diagram(ax)
    nose.tools.assert_is_instance(tmp_lines, list)


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
    invalid_params_4 = {'A0': -1.0, 'g': 0.02, 'L0': 1.0, 'n': 0.02, 's': 0.15,
                        'alpha': 0.33, 'delta': 0.03}
    invalid_params_3 = {'A0': 1.0, 'g': 0.02, 'L0': -1.0, 'n': 0.02, 's': 0.15,
                        'alpha': 0.33, 'delta': 0.03}

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

    # initial condition for A must be positive
    with nose.tools.assert_raises(AttributeError):
        solow.Model(output=valid_output, params=invalid_params_4)


def test_evaluate_output_elasticity():
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

                        actual_elasticity = tmp_mod.evaluate_output_elasticity(tmp_k_star)
                        expected_elasticity = tmp_params['alpha']

                        # conduct the test
                        nose.tools.assert_almost_equals(actual_elasticity,
                                                        expected_elasticity)
