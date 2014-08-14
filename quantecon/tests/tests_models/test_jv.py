"""
tests for quantecon.jv

@author : Spencer Lyon
@date : 2014-08-01 13:53:29

"""
from __future__ import division
import sys
import unittest
from nose.plugins.skip import SkipTest
from quantecon.models import JvWorker
from quantecon import compute_fixed_point
from quantecon.tests import get_h5_data_file, write_array, max_abs_diff

# specify params -- use defaults
A = 1.4
alpha = 0.6
beta = 0.96
grid_size = 50

if sys.version_info[0] == 2:
    v_nm = "V"
else:  # python 3
    raise SkipTest("Python 3 tests aren't ready.")
    v_nm = "V_py3"


def _new_solution(jv, f, grp):
    "gets new solution and updates data file"
    V = _solve_via_vfi(jv)
    write_array(f, grp, V, v_nm)

    return V


def _solve_via_vfi(jv):
    "compute policy rules via value function iteration"
    v_init = jv.x_grid * 0.6
    V = compute_fixed_point(jv.bellman_operator, v_init,
                            max_iter=3000,
                            error_tol=1e-5)
    return V


def _get_vf_guess(jv, force_new=False):
    with get_h5_data_file() as f:

        # See if the jv group already exists
        group_existed = True
        try:
            jv_group = f.getNode("/jv")
        except:
            # doesn't exist
            group_existed = False
            jv_group = f.create_group("/", "jv", "data for jv.py tests")

        if force_new or not group_existed:
            # group doesn't exist, or forced to create new data.
            # This function updates f in place and returns v_vfi, c_vfi, c_pfi
            V = _new_solution(jv, f, jv_group)

            return V

        # if we made it here, the group exists and we should try to read
        # existing solutions
        try:
            # Try reading vfi
            if sys.version_info[0] == 2:
                V = jv_group.V[:]
            else:  # python 3
                V = jv_group.V_py3[:]

        except:
            # doesn't exist. Let's create it
            V = _new_solution(jv, f, jv_group)

    return V


class TestJvWorkder(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        jv = JvWorker(A=A, alpha=alpha, beta=beta, grid_size=grid_size)
        cls.jv = jv

        # compute solution
        v_init = _get_vf_guess(jv)
        cls.V = compute_fixed_point(jv.bellman_operator, v_init)
        cls.s_pol, cls.phi_pol = jv.bellman_operator(cls.V * 0.999,
                                                     return_policies=True)

    def test_low_x_prefer_s(self):
        "jv: s preferred to phi with low x?"
        # low x is an early index
        self.assertGreaterEqual(self.s_pol[0], self.phi_pol[0])

    def test_high_x_prefer_phi(self):
        "jv: phi preferred to s with high x?"
        # low x is an early index
        self.assertGreaterEqual(self.phi_pol[-1], self.s_pol[-1])

    def test_policy_sizes(self):
        "jv: policies correct size"
        n = self.jv.x_grid.size
        self.assertEqual(self.s_pol.size, n)
        self.assertEqual(self.phi_pol.size, n)

    def test_bellman_sol_fixed_point(self):
        "jv: solution to bellman is fixed point"
        new_V = self.jv.bellman_operator(self.V)
        self.assertLessEqual(max_abs_diff(new_V, self.V), 1e-4)
