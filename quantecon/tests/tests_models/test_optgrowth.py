"""
tests for quantecon.models.optgrowth

@author : Spencer Lyon
@date : 2014-08-05 10:20:53

TODO: I'd really like to see why the solutions only match analytical
      counter part up to 1e-2. Seems like we should be able to do better
      than that.
"""
from __future__ import division
from math import log
import numpy as np
from nose.tools import (assert_equal, assert_true, assert_less_equal)
from quantecon import compute_fixed_point
from quantecon.models import GrowthModel
from quantecon.tests import (get_h5_data_file, get_h5_data_group, write_array,
                             max_abs_diff)


# helper parameters
_tol = 1e-6


# helper functions
def _new_solution(gm, f, grp):
    "gets a new set of solution objects and updates the data file"

    # compute value function and policy rule using vfi
    v_init = 5 * gm.u(gm.grid) - 25
    v = compute_fixed_point(gm.bellman_operator, v_init, error_tol=_tol,
                            max_iter=5000)
    # sigma = gm.get_greedy(v)

    # write all arrays to file
    write_array(f, grp, v, "v")

    # return data
    return v


def _get_data(gm, force_new=False):
    "get solution data from file, or create if necessary"
    with get_h5_data_file() as f:
        existed, grp = get_h5_data_group("optgrowth")

        if force_new or not existed:
            if existed:
                grp.w._f_remove()
            v = _new_solution(gm, f, grp)

            return v

        # if we made it here, the group exists and we should try to read
        # existing solutions
        try:
            # Try reading data
            v = grp.v[:]

        except:
            # doesn't exist. Let's create it
            v = _new_solution(gm, f, grp)

    return v

# model parameters
alpha = 0.65
f = lambda k: k ** alpha
beta = 0.95
u = np.log
grid_max = 2
grid_size = 150

gm = GrowthModel(f, beta, u, grid_max, grid_size)

v = _get_data(gm)

# compute analytical policy function
true_sigma = (1 - alpha * beta) * gm.grid**alpha

# compute analytical value function
ab = alpha * beta
c1 = (log(1 - ab) + log(ab) * ab / (1 - ab)) / (1 - beta)
c2 = alpha / (1 - ab)
def v_star(k):
    return c1 + c2 * np.log(k)


def test_h5_access():
    "optgrowth: test access to data file"
    assert_true(v is not None)


def test_bellman_return_both():
    "optgrowth: bellman_operator compute_policy option works"
    assert_equal(len(gm.bellman_operator(v, compute_policy=True)), 2)


def test_analytical_policy():
    "optgrowth: approx sigma matches analytical"
    sigma = gm.compute_greedy(v)
    assert_less_equal(max_abs_diff(sigma, true_sigma), 1e-2)


def test_analytical_vf():
    "optgrowth: approx v matches analytical"
    true_v = v_star(gm.grid)
    assert_less_equal(max_abs_diff(v[1:-1], true_v[1:-1]), 5e-2)


def test_vf_fixed_point():
    "optgrowth: solution is fixed point of bellman"
    new_v = gm.bellman_operator(v)
    assert_less_equal(max_abs_diff(v[1:-1], new_v[1:-1]), 5e-2)
