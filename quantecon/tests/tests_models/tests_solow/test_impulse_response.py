"""
Test suite for the impulse_response.py module.

@author : David R. Pugh

"""
import nose

#import matplotlib
#matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import sympy as sym

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
    original_params = {'A0': 1.0, 'g': 0.02, 'L0': 1.0, 'n': 0.02, 's': 0.15,
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


def test_plot_efficiency_units_impulse_response():
    """Testing return type for plot_impulse_response."""
    original_params = {'A0': 1.0, 'g': 0.02, 'L0': 1.0, 'n': 0.02, 's': 0.15,
                       'alpha': 0.33, 'delta': 0.05}
    model = cobb_douglas.CobbDouglasModel(original_params)

    fig, ax = plt.subplots(1, 1)
    tmp_lines = model.irf.plot_impulse_response(ax, variable='output',
                                                impulse={'g': 0.05},
                                                )
    nose.tools.assert_is_instance(tmp_lines, list)


def test_plot_levels_impulse_response():
    """Testing return type for plot_impulse_response."""
    original_params = {'A0': 1.0, 'g': 0.02, 'L0': 1.0, 'n': 0.02, 's': 0.15,
                       'alpha': 0.33, 'delta': 0.05}
    model = cobb_douglas.CobbDouglasModel(original_params)

    fig, ax = plt.subplots(1, 1)
    tmp_lines = model.irf.plot_impulse_response(ax, variable='output',
                                                impulse={'alpha': 0.25},
                                                kind='levels',
                                                log=False)
    nose.tools.assert_is_instance(tmp_lines, list)


def test_plot_per_capita_impulse_response():
    """Testing return type for plot_impulse_response."""
    original_params = {'A0': 1.0, 'g': 0.02, 'L0': 1.0, 'n': 0.02, 's': 0.15,
                       'alpha': 0.33, 'delta': 0.05}
    model = cobb_douglas.CobbDouglasModel(original_params)

    fig, ax = plt.subplots(1, 1)
    tmp_lines = model.irf.plot_impulse_response(ax, variable='output',
                                                impulse={'s': 0.5},
                                                kind='per_capita',
                                                log=True)
    nose.tools.assert_is_instance(tmp_lines, list)


def test_valid_kind():
    """Testing validation of the kind attribute."""
    # kind attribute must be a string
    with nose.tools.assert_raises(AttributeError):
        model.irf.kind = 5

    # kind sttribute must be a valid string
    with nose.tools.assert_raises(AttributeError):
        model.irf.kind = 'invalid_kind'
