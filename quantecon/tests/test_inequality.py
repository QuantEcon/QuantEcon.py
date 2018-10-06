
"""
Tests for inequality.py

"""

import numpy as np
from numpy.testing import assert_allclose
from quantecon import lorenz_curve, gini_coefficient


def test_lorenz_curve():
    """
    Tests `lorenz` function, which calculates the lorenz curve

    An income distribution where everyone has almost the same wealth should
    be similar to a straight line

    An income distribution where one person has almost the wealth should
    be flat and then shoot straight up when it approaches one
    """
    n = 3000

    # Almost Equal distribution
    y = np.repeat(1, n) + np.random.normal(scale=0.0001, size=n)
    cum_people, cum_income = lorenz_curve(y)
    assert_allclose(cum_people, cum_income, rtol=1e-03)

    # Very uneven distribution
    y = np.repeat(0.001, n)
    y[4] = 100000
    pop_cum, income_cum = lorenz_curve(y)
    expected_income_cum = np.repeat(0., n + 1)
    expected_income_cum[-1] = 1.
    assert_allclose(expected_income_cum, income_cum, atol=1e-4)


def test_gini_coeff():
    """
    Tests how the funciton `gini_coefficient` calculates the Gini coefficient
    with the Pareto and the Weibull distribution.

    Analytically, we know that Pareto with parameter `a` has
    G = 1 / (2*a - 1)

    Likewise, for the Weibull distribution with parameter `a` we know that
    G = 1 - 2**(-1/a)

    """
    n = 10000

    # Tests Pareto: G = 1 / (2*a - 1)
    a = np.random.randint(2, 15)
    expected = 1 / (2 * a - 1)

    y = (np.random.pareto(a, size=n) + 1) * 2
    coeff = gini_coefficient(y)
    assert_allclose(expected, coeff, rtol=1e-01)

    # Tests Weibull: G = 1 - 2**(-1/a)
    a = np.random.randint(2, 15)
    expected = 1 - 2 ** (-1 / a)

    y = np.random.weibull(a, size=n)
    coeff = gini_coefficient(y)
    assert_allclose(expected, coeff, rtol=1e-01)