"""
Test suite for solow module.

@author : David R. Pugh
@date : 2014-09-01

"""
import nose

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

# four different ways in which params can fail
invalid_params_0 = (0.02, 0.02, 0.15, 0.33, 0.03)
invalid_params_1 = {'g': -0.02, 'n': -0.02, 's': 0.15, 'alpha': 0.33,
                    'delta': 0.03}
invalid_params_2 = {'g': 0.02, 'n': 0.02, 's': 0.15, 'alpha': 0.33,
                    'delta': -0.03}
invalid_params_3 = {'g': 0.02, 'n': 0.02, 's': -0.15, 'alpha': 0.33,
                    'delta': 0.03}

valid_params = {'g': 0.02, 'n': 0.02, 's': 0.15, 'alpha': 0.33, 'delta': 0.05}


# helper functions
def k_upper(model):
    """Upper bound on steady state value for model with Cobb Douglas production."""
    alpha = model.params['alpha']
    return (1 / model.effective_depreciation_rate)**(1 / (1 - alpha))


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

                        tmp_params = {'g': g, 'n': n, 's': s, 'alpha': alpha,
                                      'delta': delta}
                        tmp_mod = solow.Model(output=valid_output,
                                              params=tmp_params)

                        # use root finder to compute the steady state
                        tmp_k_upper = k_upper(tmp_mod)
                        actual_ss = tmp_mod.find_steady_state(1e-12, tmp_k_upper)
                        expected_ss = cobb_douglas.analytic_steady_state(tmp_mod)

                        # conduct the test (default places=7 is too precise!)
                        nose.tools.assert_almost_equals(actual_ss, expected_ss,
                                                        places=6)


def test_ivp_solve():
    """Testing computation of solution to the initial value problem."""
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
                        tmp_k_upper = k_upper(tmp_mod)
                        tmp_k_star = tmp_mod.find_steady_state(1e-12, tmp_k_upper)

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
        tmp_k_upper = k_upper(tmp_mod)
        actual_ss = tmp_mod.find_steady_state(1e-12, tmp_k_upper, method=method)
        expected_ss = cobb_douglas.analytic_steady_state(tmp_mod)
        nose.tools.assert_almost_equals(actual_ss, expected_ss)


def test_valid_methods():
    """Testing raise exception if invalid method passed to find_steady_state."""
    with nose.tools.assert_raises(ValueError):
        tmp_mod = solow.Model(output=valid_output, params=valid_params)
        tmp_mod.find_steady_state(1e-12, k_upper(tmp_mod),
                                  method='invalid_method')


def test_plot_intensive_output():
    """Testing return type for plot_intensive_output."""
    tmp_mod = solow.Model(output=valid_output, params=valid_params)
    tmp_plot = solow.model.plot_intensive_output(tmp_mod)

    # test the return types
    fig, ax = tmp_plot
    nose.tools.assert_is_instance(tmp_plot, list)
    nose.tools.assert_is_instance(fig, plt.Figure)
    nose.tools.assert_is_instance(ax, plt.Axes)


def test_plot_intensive_investment():
    """Testing return type for plot_intensive_investment."""
    tmp_mod = solow.Model(output=valid_output, params=valid_params)
    tmp_plot = solow.model.plot_intensive_investment(tmp_mod)

    # test the return types
    fig, ax = tmp_plot
    nose.tools.assert_is_instance(tmp_plot, list)
    nose.tools.assert_is_instance(fig, plt.Figure)
    nose.tools.assert_is_instance(ax, plt.Axes)


def test_plot_phase_diagram():
    """Testing return type for plot_phase_diagram."""
    tmp_mod = solow.Model(output=valid_output, params=valid_params)
    tmp_plot = solow.model.plot_phase_diagram(tmp_mod)

    # test the return types
    fig, ax = tmp_plot
    nose.tools.assert_is_instance(tmp_plot, list)
    nose.tools.assert_is_instance(fig, plt.Figure)
    nose.tools.assert_is_instance(ax, plt.Axes)


def test_plot_solow_diagram():
    """Testing return type for plot_solow_diagram."""
    tmp_mod = solow.Model(output=valid_output, params=valid_params)
    tmp_plot = solow.model.plot_solow_diagram(tmp_mod)

    # test the return types
    fig, ax = tmp_plot
    nose.tools.assert_is_instance(tmp_plot, list)
    nose.tools.assert_is_instance(fig, plt.Figure)
    nose.tools.assert_is_instance(ax, plt.Axes)
