"""
Test suite for the impulse_response.py module.

@author : David R. Pugh

"""
from __future__ import division
import nose

import matplotlib.pyplot as plt
import numpy as np

from .... models.solow import cobb_douglas

params = {'A0': 1.0, 'g': 0.02, 'L0': 1.0, 'n': 0.02, 's': 0.15,
          'alpha': 0.33, 'delta': 0.05}
model = cobb_douglas.CobbDouglasModel(params)


def test_valid_impulse():
    """Testing validation of impulse attribute."""
    # impulse attribute must be a dict
    with nose.tools.assert_raises(AttributeError):
        model.irf.impulse = (('alpha', 0.75), ('g', 0.04))

    # impulse sttribute must have valid keys
    with nose.tools.assert_raises(AttributeError):
        model.irf.impulse = {'alpha': 0.56, 'bad_key': 0.55}


def test_impulse_response():
    """Testing computation of impulse response."""
    original_params = {'A0': 1.0, 'g': 0.01, 'L0': 1.0, 'n': 0.02, 's': 0.15,
                       'alpha': 0.33, 'delta': 0.05}
    model = cobb_douglas.CobbDouglasModel(original_params)

    # generate the impulse response
    impulse = {'s': 0.30}
    model.irf.impulse = impulse
    model.irf.kind = 'efficiency_units'
    model.irf.T = 500  # need to get "close" to new BGP
    actual_ss = model.irf.impulse_response[-1, 1]

    # compute steady state following the impulse
    model.params.update(impulse)
    expected_ss = model.steady_state

    nose.tools.assert_almost_equals(actual_ss, expected_ss)


def test_per_capita_impulse_response():
    """Testing computation of per capita impulse response."""
    original_params = {'A0': 1.0, 'g': 0.01, 'L0': 1.0, 'n': 0.02, 's': 0.15,
                       'alpha': 0.33, 'delta': 0.05}
    model = cobb_douglas.CobbDouglasModel(original_params)

    # generate the per capita impulse response
    impulse = {'alpha': 0.15}
    model.irf.impulse = impulse
    model.irf.kind = 'per_capita'
    model.irf.T = 500  # need to get "close" to new BGP
    actual_c = model.irf.impulse_response[-1, 3]

    # compute steady state following the impulse
    model.params.update(impulse)
    A0, g = model.params['A0'], model.params['g']
    scaling_factor = A0 * np.exp(g * model.irf.T)
    c_ss = model.evaluate_consumption(model.steady_state)
    expected_c = c_ss * scaling_factor

    nose.tools.assert_almost_equals(actual_c, expected_c)


def test_levels_impulse_response():
    """Testing computation of levels impulse response."""
    original_params = {'A0': 1.0, 'g': 0.01, 'L0': 1.0, 'n': 0.02, 's': 0.15,
                       'alpha': 0.33, 'delta': 0.05}
    model = cobb_douglas.CobbDouglasModel(original_params)

    # generate the per capita impulse response
    impulse = {'delta': 0.15}
    model.irf.impulse = impulse
    model.irf.kind = 'levels'
    model.irf.T = 500  # need to get "close" to new BGP
    actual_y = model.irf.impulse_response[-1, 2]

    # compute steady state following the impulse
    model.params.update(impulse)
    A0, g = model.params['A0'], model.params['g']
    L0, n = model.params['L0'], model.params['n']
    scaling_factor = A0 * L0 * np.exp((g + n) * model.irf.T)
    y_ss = model.evaluate_intensive_output(model.steady_state)
    expected_y = y_ss * scaling_factor

    nose.tools.assert_almost_equals(actual_y, expected_y)


def test_plot_efficiency_units_impulse_response():
    """Testing return type for plot_impulse_response."""
    original_params = {'A0': 1.0, 'g': 0.02, 'L0': 1.0, 'n': 0.02, 's': 0.15,
                       'alpha': 0.33, 'delta': 0.05}
    model = cobb_douglas.CobbDouglasModel(original_params)

    # initialize the impulse
    model.irf.impulse = {'delta': 0.25}
    model.irf.kind = 'efficiency_units'

    fig, ax = plt.subplots(1, 1)
    tmp_lines = model.irf.plot_impulse_response(ax, variable='output')
    nose.tools.assert_is_instance(tmp_lines, list)


def test_plot_levels_impulse_response():
    """Testing return type for plot_impulse_response."""
    original_params = {'A0': 1.0, 'g': 0.02, 'L0': 1.0, 'n': 0.02, 's': 0.15,
                       'alpha': 0.33, 'delta': 0.05}
    model = cobb_douglas.CobbDouglasModel(original_params)

    # initialize the impulse
    model.irf.impulse = {'alpha': 0.25}
    model.irf.kind = 'levels'

    fig, ax = plt.subplots(1, 1)
    tmp_lines = model.irf.plot_impulse_response(ax, variable='output',
                                                log=False)
    nose.tools.assert_is_instance(tmp_lines, list)


def test_plot_per_capita_impulse_response():
    """Testing return type for plot_impulse_response."""
    original_params = {'A0': 1.0, 'g': 0.02, 'L0': 1.0, 'n': 0.02, 's': 0.15,
                       'alpha': 0.33, 'delta': 0.05}
    model = cobb_douglas.CobbDouglasModel(original_params)

    # initialize the impulse
    model.irf.impulse = {'g': 0.05}
    model.irf.kind = 'per_capita'

    fig, ax = plt.subplots(1, 1)
    tmp_lines = model.irf.plot_impulse_response(ax, variable='output',
                                                log=True)
    nose.tools.assert_is_instance(tmp_lines, list)


def test_valid_kind():
    """Testing validation of the kind attribute."""

    # kind sttribute must be a valid string
    with nose.tools.assert_raises(AttributeError):
        model.irf.kind = 'invalid_kind'
