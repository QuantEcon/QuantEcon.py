"""
tests for quantecon.ifp

@author : Spencer Lyon
@date : 2014-08-01 12:09:17

"""
from __future__ import division
import unittest
import numpy as np
from quantecon.models import ConsumerProblem
from quantecon import compute_fixed_point
from quantecon.tests import get_h5_data_file, write_array, max_abs_diff


def _solve_via_vfi(cp, v_init, return_both=False):
    "compute policy rule using value function iteration"
    v = compute_fixed_point(cp.bellman_operator, v_init, verbose=False,
                            error_tol=1e-5,
                            max_iter=1000)

    # Run one more time to get the policy
    p = cp.bellman_operator(v, return_policy=True)

    if return_both:
        return v, p
    else:
        return p


def _solve_via_pfi(cp, c_init):
    "compute policy rule using policy function iteration"
    p = compute_fixed_point(cp.coleman_operator, c_init, verbose=False,
                            error_tol=1e-5,
                            max_iter=1000)

    return p


def _get_vfi_pfi_guesses(cp, force_new=False):
    """
    load precomputed vfi/pfi solutions, or compute them if requested
    or we can't find old ones
    """
    # open the data file
    with get_h5_data_file() as f:

        # See if the ifp group already exists
        group_existed = True
        try:
            ifp_group = f.getNode("/ifp")
        except:
            # doesn't exist
            group_existed = False
            ifp_group = f.create_group("/", "ifp", "data for ifp.py tests")

        if force_new or not group_existed:
            # group doesn't exist, or forced to create new data.
            # This function updates f in place and returns v_vfi, c_vfi, c_pfi
            v_vfi, c_vfi, c_pfi = _new_solutions(cp, f, ifp_group)

            # We have what we need, so return
            return v_vfi, c_pfi

        # if we made it here, the group exists and we should try to read
        # existing solutions
        try:  # read in vfi
            # Try reading vfi
            c_vfi = ifp_group.c_vfi[:]
            v_vfi = ifp_group.v_vfi[:]

        except:
            # doesn't exist. Let's create it
            v_vfi, c_vfi = _new_solutions(cp, f, ifp_group, which="vfi")

        try:  # read in pfi
            # Try reading pfi
            c_pfi = ifp_group.c_pfi[:]

        except:
            # doesn't exist. Let's create it
            c_pfi = _new_solutions(cp, f, ifp_group, which="pfi")

    return v_vfi, c_pfi


def _new_solutions(cp, f, grp, which="both"):
    v_init, c_init = cp.initialize()
    if which == "both":

        v_vfi, c_vfi = _solve_via_vfi(cp, v_init, return_both=True)
        c_pfi = _solve_via_pfi(cp, c_init)

        # Store solutions in chunked arrays...
        write_array(f, grp, c_vfi, "c_vfi")
        write_array(f, grp, v_vfi, "v_vfi")
        write_array(f, grp, c_pfi, "c_pfi")

        return v_vfi, c_vfi, c_pfi

    elif which == "vfi":
        v_vfi, c_vfi = _solve_via_vfi(cp, v_init, return_both=True)
        write_array(f, grp, c_vfi, "c_vfi")
        write_array(f, grp, v_vfi, "v_vfi")

        return v_vfi, c_vfi

    elif which == "pfi":
        c_pfi = _solve_via_pfi(cp, c_init)
        write_array(f, grp, c_pfi, "c_pfi")

        return c_pfi


class TestConsumerProblem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cp = ConsumerProblem()

        # get precomputed answer for each method
        print("reading old solutions")
        old_vfi_sol, old_pfi_sol = _get_vfi_pfi_guesses(cls.cp)
        cls.v_vfi = old_vfi_sol

        # compute answers again, using something close to old answer as
        # initial value so it goes really fast
        print("computing new vfi")
        cls.c_vfi = _solve_via_vfi(cls.cp, old_vfi_sol * 0.99999)
        print("computing new pfi")
        cls.c_pfi = _solve_via_pfi(cls.cp, old_pfi_sol * 0.99999)

    def test_bellman_coleman_solutions_agree(self):
        "ifp: bellman and coleman solutions agree"
        self.assertLessEqual(max_abs_diff(self.c_vfi, self.c_pfi), 0.2)

    def test_bellman_fp(self):
        "ifp: solution to bellman is a fixed point"
        new_v = self.cp.bellman_operator(self.v_vfi)
        self.assertLessEqual(max_abs_diff(self.v_vfi, new_v), 1e-3)

    def test_coleman_fp(self):
        "ifp: solution to coleman is a fixed point"
        new_c = self.cp.coleman_operator(self.c_pfi)
        self.assertLessEqual(max_abs_diff(self.c_pfi, new_c), 1e-3)
