"""
Tests for quantecon.models.lucastree

@author : Spencer Lyon
@date : 2014-08-05 09:15:45

"""
from __future__ import division
from nose.tools import (assert_equal, assert_true, assert_less_equal)
import numpy as np
from quantecon.models import LucasTree
from quantecon.tests import (get_h5_data_file, get_h5_data_group, write_array,
                             max_abs_diff)

# helper parameters
_tol = 1e-6


# helper functions
def _new_solution(tree, f, grp):
    "gets a new set of prices and updates the file"
    prices = tree.compute_lt_price(error_tol=_tol, max_iter=5000)
    write_array(f, grp, prices, "prices")
    return prices


def _get_price_data(tree, force_new=False):
    "get price data from file, or create if necessary"
    with get_h5_data_file() as f:
        existed, grp = get_h5_data_group("lucastree")

        if force_new or not existed:
            if existed:
                grp.prices._f_remove()
            prices = _new_solution(tree, f, grp)

            return prices

        # if we made it here, the group exists and we should try to read
        # existing solutions
        try:
            # Try reading vfi
            prices = grp.prices[:]

        except:
            # doesn't exist. Let's create it
            prices = _new_solution(tree, f, grp)

    return prices


# model parameters
gamma = 2.0
beta = 0.95
alpha = 0.90
sigma = 0.1

# model object
tree = LucasTree(gamma, beta, alpha, sigma)
grid = tree.grid
prices = _get_price_data(tree)


def test_h5_access():
    "lucastree: test access to data file"
    assert_true(prices is not None)


def test_prices_shape():
    "lucastree: test access shape of computed prices"
    assert_equal(prices.shape, grid.shape)


def test_integrate():
    "lucastree: integrate function"
    # just have it be a 1. Then integrate should give cdf
    g = lambda x: x*0.0 + 1.0

    # estimate using integrate function
    est = tree.integrate(g)

    # compute exact solution
    exact = tree.phi.cdf(tree._int_max) - tree.phi.cdf(tree._int_min)

    assert_less_equal(est - exact, .1)


def test_lucas_op_fixed_point():
    "lucastree: are prices a fixed point of lucas_operator"
    # transform from p to f
    old_f = prices / (grid ** gamma)

    # compute new f
    new_f = tree.lucas_operator(old_f)

    # transform from f to p
    new_p = new_f * grid**gamma

    # test if close. Make it one order of magnitude less than tol used
    # to compute prices
    assert_less_equal(max_abs_diff(new_p, prices), _tol*10)


def test_lucas_prices_increasing():
    "lucastree: test prices are increasing in y"
    # sort the array and test that it is the same
    sorted = np.sort(np.copy(prices))
    np.testing.assert_array_equal(sorted, prices)
