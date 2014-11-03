"""
tests for quantecon.models.odu

@author : Spencer Lyon
@date : 2014-08-05 10:20:53

"""
from __future__ import division
import numpy as np
from nose.tools import (assert_equal, assert_true, assert_less_equal)
from quantecon import compute_fixed_point
from quantecon.models import SearchProblem
from quantecon.tests import (get_h5_data_file, get_h5_data_group, write_array,
                             max_abs_diff)

# helper parameters
_tol = 1e-6


# helper functions
def _new_solution(sp, f, grp):
    "gets a new set of solution objects and updates the data file"

    # compute value function and policy rule using vfi
    v_init = np.zeros(len(sp.grid_points)) + sp.c / (1 - sp.beta)
    v = compute_fixed_point(sp.bellman_operator, v_init, error_tol=_tol,
                            max_iter=5000)
    phi_vfi = sp.get_greedy(v)

    # also run v through bellman so I can test if it is a fixed point
    # bellman_operator takes a long time, so store result instead of compute
    new_v = sp.bellman_operator(v)

    # compute policy rule using pfi

    phi_init = np.ones(len(sp.pi_grid))
    phi_pfi = compute_fixed_point(sp.res_wage_operator, phi_init,
                                  error_tol=_tol, max_iter=5000)

    # write all arrays to file
    write_array(f, grp, v, "v")
    write_array(f, grp, phi_vfi, "phi_vfi")
    write_array(f, grp, phi_pfi, "phi_pfi")
    write_array(f, grp, new_v, "new_v")

    # return data
    return v, phi_vfi, phi_pfi, new_v


def _get_data(sp, force_new=False):
    "get solution data from file, or create if necessary"
    with get_h5_data_file() as f:
        existed, grp = get_h5_data_group("odu")

        if force_new or not existed:
            if existed:
                grp.v._f_remove()
                grp.phi_vfi._f_remove()
                grp.phi_pfi._f_remove()
                grp.new_v._f_remove()
            v, phi_vfi, phi_pfi, new_v = _new_solution(sp, f, grp)

            return v, phi_vfi, phi_pfi, new_v

        # if we made it here, the group exists and we should try to read
        # existing solutions
        try:
            # Try reading data
            v = grp.v[:]
            phi_vfi = grp.phi_vfi[:]
            phi_pfi = grp.phi_pfi[:]
            new_v = grp.new_v[:]

        except:
            # doesn't exist. Let's create it
            v, phi_vfi, phi_pfi, new_v = _new_solution(sp, f, grp)

    return v, phi_vfi, phi_pfi, new_v

# model parameters
beta = 0.95
c = 0.6
F_a = 1
F_b = 1
G_a = 3
G_b = 1.2
w_max = 2
w_grid_size = 40
pi_grid_size = 40

sp = SearchProblem(beta, c, F_a, F_b, G_a, G_b, w_max, w_grid_size,
                   pi_grid_size)

v, phi_vfi, phi_pfi, new_v = _get_data(sp)


def test_h5_access():
    "odu: test access to data file"
    assert_true(v is not None)
    assert_true(phi_vfi is not None)
    assert_true(phi_pfi is not None)


def test_vfi_v_phi_same_shape():
    "odu: vfi value and policy same shape"
    assert_equal(v.shape, phi_vfi.shape)


def test_phi_vfi_increasing():
    "odu: phi from vfi is increasing"
    phi_mat = phi_vfi.reshape(w_grid_size, pi_grid_size)
    sorted = np.sort(np.copy(phi_mat))
    np.testing.assert_array_equal(sorted, phi_mat)


def test_phi_pfi_increasing():
    "odu: phi from pfi is increasing"
    sorted = np.sort(np.copy(phi_pfi))[::-1]  # ascending
    np.testing.assert_array_equal(sorted, phi_pfi)


def test_v_vfi_increasing():
    "odu: v from vfi is increasing"
    # order so it sorts along the correct dimension (ascending)
    v_mat = v[::-1].reshape(w_grid_size, pi_grid_size)
    sorted = np.sort(np.copy(v_mat))
    np.testing.assert_array_equal(sorted, v_mat)


def test_v_vfi_fixed_point():
    "odu: v from vfi is fixed point"
    assert_less_equal(max_abs_diff(v, new_v), _tol*10)


def test_phi_pfi_fixed_point():
    "odu: phi from pfi is fixed point"
    new_phi = sp.res_wage_operator(phi_pfi)
    assert_less_equal(max_abs_diff(new_phi, phi_pfi), _tol*10)
