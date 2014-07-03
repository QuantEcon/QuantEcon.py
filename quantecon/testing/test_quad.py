"""
Filename: test_quad.py
Authors: Chase Coleman, Spencer Lyon, John Stachurski, Thomas Sargent
Date: 2014-07-02

Tests for quantecon.quad module

Notes
-----
Many of tests were derived from the file demqua03 in the CompEcon
toolbox.

For all other tests, the MATLAB code is provided here in
a section of comments.

"""
from __future__ import division
import os
import unittest
from scipy.io import loadmat
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
from quantecon.quad import *

### MATLAB code needed to generate data (in addition to a modified demqua03)
# % set random number seed so we get the same random nums as in python
# rng(42)
# % 1-d parameters -- just some random numbers
# a = -2.0
# b = 3.0
# n = 11

# % 3-d parameters -- just some random numbers
# a_3 = [-1.0 -2.0 1.0]
# b_3 = [1.0 12.0 1.5]
# n_3 = [7 5 9]

# mu_3d = [1.0 2.0 2.5]
# sigma2_3d = [1.0 0.1 0.0; 0.1 1.0 0.0; 0.0 0.0 1.2]

# % 1-d nodes and weights
# [x_cheb_1 w_cheb_1] = qnwcheb(n, a, b)
# [x_equiN_1 w_equiN_1] = qnwequi(n, a, b, 'N')
# [x_equiW_1 w_equiW_1] = qnwequi(n, a, b, 'W')
# [x_equiH_1 w_equiH_1] = qnwequi(n, a, b, 'H')
# rng(41); [x_equiR_1 w_equiR_1] = qnwequi(n, a, b, 'R')
# [x_lege_1 w_lege_1] = qnwlege(n, a, b)
# [x_norm_1 w_norm_1] = qnwnorm(n, a, b)
# [x_logn_1 w_logn_1] = qnwlogn(n, a, b)
# [x_simp_1 w_simp_1] = qnwsimp(n, a, b)
# [x_trap_1 w_trap_1] = qnwtrap(n, a, b)
# [x_unif_1 w_unif_1] = qnwunif(n, a, b)
# [x_beta_1 w_beta_1] = qnwbeta(n, b, b+1)
# [x_gamm_1 w_gamm_1] = qnwgamma(n, b)

# % 3-d nodes and weights
# [x_cheb_3 w_cheb_3] = qnwcheb(n_3, a_3, b_3)
# rng(42); [x_equiN_3 w_equiN_3] = qnwequi(n_3, a_3, b_3, 'N')
# [x_equiW_3 w_equiW_3] = qnwequi(n_3, a_3, b_3, 'W')
# [x_equiH_3 w_equiH_3] = qnwequi(n_3, a_3, b_3, 'H')
# [x_equiR_3 w_equiR_3] = qnwequi(n_3, a_3, b_3, 'R')
# [x_lege_3 w_lege_3] = qnwlege(n_3, a_3, b_3)
# [x_norm_3 w_norm_3] = qnwnorm(n_3, mu_3d, sigma2_3d)
# [x_logn_3 w_logn_3] = qnwlogn(n_3, mu_3d, sigma2_3d)
# [x_simp_3 w_simp_3] = qnwsimp(n_3, a_3, b_3)
# [x_trap_3 w_trap_3] = qnwtrap(n_3, a_3, b_3)
# [x_unif_3 w_unif_3] = qnwunif(n_3, a_3, b_3)
# [x_beta_3 w_beta_3] = qnwbeta(n_3, b_3, b_3+1.0)
# [x_gamm_3 w_gamm_3] = qnwgamma(n_3, b_3)

### End MATLAB commands

this_dir = os.path.dirname(__file__)
# this_dir = os.path.abspath(".")
data_dir = os.path.join(this_dir, "data")
data = loadmat(os.path.join(data_dir, "matlab_quad.mat"), squeeze_me=True)

# Unpack parameters from MATLAB
a = data['a']
b = data['b']
n = data['n']
a_3 = data['a_3']
b_3 = data['b_3']
n_3 = data['n_3']
mu_3d = data['mu_3d']
sigma2_3d = data['sigma2_3d']


class TestQuadrect(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        ## Create Python Data for quadrect
        # Create the python data -- similar to notebook code
        kinds = ["trap", "simp", "lege", "N", "W", "H", "R"]

        # Define some functions
        f1 = lambda x: np.exp(-x)
        f2 = lambda x: 1.0 / (1.0 + 25.0 * x**2.0)
        f3 = lambda x: np.abs(x) ** 0.5
        func_names = ["f1", "f2", "f3"]

        # Integration parameters
        n = np.array([5, 11, 21, 51, 101, 401])  # number of nodes
        np.random.seed(42)  # same seed as ML code.
        a, b = -1, 1  # endpoints

        # Set up pandas DataFrame to hold results
        ind = pd.MultiIndex.from_product([func_names, n])
        ind.names = ["Function", "Number of Nodes"]
        cols = pd.Index(kinds, name="Kind")
        quad_rect_res1d = pd.DataFrame(index=ind, columns=cols, dtype=float)

        for i, func in enumerate([f1, f2, f3]):
            func_name = func_names[i]
            for kind in kinds:
                for num in n:
                    num_in = num ** 2 if len(kind) == 1 else num
                    quad_rect_res1d.ix[func_name, num][kind] = quadrect(func,
                                                                        num_in,
                                                                        a, b,
                                                                        kind)

        cls.data1d = quad_rect_res1d

        # Now 2d data
        kinds2 = ["lege", "trap", "simp", "N", "W", "H", "R"]
        f1_2 = lambda x: np.exp(x[:, 0] + x[:, 1])
        f2_2 = lambda x: np.exp(-x[:, 0] * np.cos(x[:, 1]**2))

        # Set up pandas DataFrame to hold results
        a = ([0, 0], [-1, -1])
        b = ([1, 2], [1, 1])
        ind_2 = pd.Index(n**2, name="Num Points")
        cols2 = pd.Index(kinds2, name="Kind")
        data2 = pd.DataFrame(index=ind_2, columns=cols2, dtype=float)

        for num in n:
            for kind in kinds2[:4]:
                data2.ix[num**2][kind] = quadrect(f1_2, [num, num],
                                                  a[0], b[0], kind)
            for kind in kinds2[4:]:
                data2.ix[num**2][kind] = quadrect(f1_2, num**2, a[0],
                                                  b[0], kind)

        cls.data2d1 = data2

        n3 = 10 ** (2 + np.array([1, 2, 3]))
        ind_3 = pd.Index(n3, name="Num Points")
        cols3 = pd.Index(kinds2[3:])
        data3 = pd.DataFrame(index=ind_3, columns=cols3, dtype=float)

        for num in n3:
            for kind in kinds2[3:]:
                data3.ix[num][kind] = quadrect(f2_2, num, a[1], b[1], kind)

        cls.data2d2 = data3

        ## Organize MATLAB Data
        ml_data = pd.DataFrame(index=ind, columns=cols, dtype=float)

        ml_data.iloc[:6, :] = data['int_1d'][:, :, 0]
        ml_data.iloc[6:12, :] = data['int_1d'][:, :, 1]
        ml_data.iloc[12:18, :] = data['int_1d'][:, :, 2]

        ml_data2 = pd.DataFrame(index=ind_2, columns=cols2, dtype=float)
        ml_data2.iloc[:, :] = data['int_2d1']

        ml_data3 = pd.DataFrame(index=ind_3, columns=cols3, dtype=float)
        ml_data3.iloc[:, :] = data['int_2d2']

        cls.ml_data1d = ml_data
        cls.ml_data2d1 = ml_data2
        cls.ml_data2d2 = ml_data3

    def test_quadrect_1d_lege(self):
        assert_allclose(self.data1d['lege'], self.ml_data1d['lege'])

    def test_quadrect_1d_trap(self):
        assert_allclose(self.data1d['trap'], self.ml_data1d['trap'])

    def test_quadrect_1d_simp(self):
        assert_allclose(self.data1d['simp'], self.ml_data1d['simp'])

    def test_quadrect_1d_R(self):
        assert_allclose(self.data1d['R'], self.ml_data1d['R'])

    def test_quadrect_1d_W(self):
        assert_allclose(self.data1d['W'], self.ml_data1d['W'])

    def test_quadrect_1d_N(self):
        assert_allclose(self.data1d['N'], self.ml_data1d['N'])

    def test_quadrect_1d_H(self):
        assert_allclose(self.data1d['H'], self.ml_data1d['H'])

    def test_quadrect_2d_lege(self):
        assert_allclose(self.data2d1['lege'], self.ml_data2d1['lege'])

    def test_quadrect_2d_trap(self):
        assert_allclose(self.data2d1['trap'], self.ml_data2d1['trap'])

    def test_quadrect_2d_simp(self):
        assert_allclose(self.data2d1['simp'], self.ml_data2d1['simp'])

    # NOTE: The R tests will fail in more than 1 dimension. This is a
    #       function of MATLAB and numpy storing arrays in different
    #       "order". See comment in TestQnwequiR.setUpClass
    # def test_quadrect_2d_R(self):
    #     assert_allclose(self.data2d1['R'], self.ml_data2d1['R'])

    def test_quadrect_2d_W(self):
        assert_allclose(self.data2d1['W'], self.ml_data2d1['W'])

    def test_quadrect_2d_N(self):
        assert_allclose(self.data2d1['N'], self.ml_data2d1['N'])

    def test_quadrect_2d_H(self):
        assert_allclose(self.data2d1['H'], self.ml_data2d1['H'])

    def test_quadrect_2d_W2(self):
        assert_allclose(self.data2d2['W'], self.ml_data2d2['W'])

    def test_quadrect_2d_N2(self):
        assert_allclose(self.data2d2['N'], self.ml_data2d2['N'])

    def test_quadrect_2d_H2(self):
        assert_allclose(self.data2d2['H'], self.ml_data2d2['H'])


class TestQnwcheb(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.x_cheb_1, cls.w_cheb_1 = qnwcheb(n, a, b)
        cls.x_cheb_3, cls.w_cheb_3 = qnwcheb(n_3, a_3, b_3)

    def test_qnwcheb_nodes_1d(self):
        assert_allclose(self.x_cheb_1, data['x_cheb_1'])

    def test_qnwcheb_nodes_3d(self):
        assert_allclose(self.x_cheb_3, data['x_cheb_3'])

    def test_qnwcheb_weights_1d(self):
        assert_allclose(self.w_cheb_1, data['w_cheb_1'])

    def test_qnwcheb_weights_3d(self):
        assert_allclose(self.w_cheb_3, data['w_cheb_3'])


class TestQnwequiN(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.x_equiN_1, cls.w_equiN_1 = qnwequi(n, a, b, "N")
        cls.x_equiN_3, cls.w_equiN_3 = qnwequi(n_3, a_3, b_3, "N")

    def test_qnwequiN_nodes_1d(self):
        assert_allclose(self.x_equiN_1, data['x_equiN_1'])

    def test_qnwequiN_nodes_3d(self):
        assert_allclose(self.x_equiN_3, data['x_equiN_3'])

    def test_qnwequiN_weights_1d(self):
        assert_allclose(self.w_equiN_1, data['w_equiN_1'])

    def test_qnwequiN_weights_3d(self):
        assert_allclose(self.w_equiN_3, data['w_equiN_3'])


class TestQnwequiW(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.x_equiW_1, cls.w_equiW_1 = qnwequi(n, a, b, "W")
        cls.x_equiW_3, cls.w_equiW_3 = qnwequi(n_3, a_3, b_3, "W")

    def test_qnwequiW_nodes_1d(self):
        assert_allclose(self.x_equiW_1, data['x_equiW_1'])

    def test_qnwequiW_nodes_3d(self):
        assert_allclose(self.x_equiW_3, data['x_equiW_3'])

    def test_qnwequiW_weights_1d(self):
        assert_allclose(self.w_equiW_1, data['w_equiW_1'])

    def test_qnwequiW_weights_3d(self):
        assert_allclose(self.w_equiW_3, data['w_equiW_3'])


class TestQnwequiH(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.x_equiH_1, cls.w_equiH_1 = qnwequi(n, a, b, "H")
        cls.x_equiH_3, cls.w_equiH_3 = qnwequi(n_3, a_3, b_3, "H")

    def test_qnwequiH_nodes_1d(self):
        assert_allclose(self.x_equiH_1, data['x_equiH_1'])

    def test_qnwequiH_nodes_3d(self):
        assert_allclose(self.x_equiH_3, data['x_equiH_3'])

    def test_qnwequiH_weights_1d(self):
        assert_allclose(self.w_equiH_1, data['w_equiH_1'])

    def test_qnwequiH_weights_3d(self):
        assert_allclose(self.w_equiH_3, data['w_equiH_3'])


class TestQnwequiR(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        np.random.seed(41)  # make sure to set seed here.
        cls.x_equiR_1, cls.w_equiR_1 = qnwequi(n, a, b, "R")
        np.random.seed(42)  # make sure to set seed here.
        temp, cls.w_equiR_3 = qnwequi(n_3, a_3, b_3, "R")

        # NOTE: I need to do a little magic here. MATLAB and numpy
        #       are generating the same random numbers, but MATLAB is
        #       column major and numpy is row major, so they are stored
        #       in different places for multi-dimensional arrays.
        #       The ravel, reshape code here moves the numpy nodes into
        #       the same relative position as the MATLAB ones. Also, in
        #       order for this to work I have to undo the shifting of
        #       the nodes, re-organize data, then re-shift. If this
        #       seems like voodoo to you, it kinda is. But, the fact
        #       that the test can pass after this kind of manipulation
        #       is a strong indicator that we are doing it correctly

        unshifted = (temp - a_3) / (b_3 - a_3)
        reshaped = np.ravel(unshifted).reshape(315, 3, order='F')
        reshifted = a_3 + reshaped * (b_3 - a_3)
        cls.x_equiR_3 = reshifted

    def test_qnwequiR_nodes_1d(self):
        assert_allclose(self.x_equiR_1, data['x_equiR_1'])

    def test_qnwequiR_nodes_3d(self):
        assert_allclose(self.x_equiR_3, data['x_equiR_3'])

    def test_qnwequiR_weights_1d(self):
        assert_allclose(self.w_equiR_1, data['w_equiR_1'])

    def test_qnwequiR_weights_3d(self):
        assert_allclose(self.w_equiR_3, data['w_equiR_3'])


class TestQnwlege(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.x_lege_1, cls.w_lege_1 = qnwlege(n, a, b)
        cls.x_lege_3, cls.w_lege_3 = qnwlege(n_3, a_3, b_3)

    def test_qnwlege_nodes_1d(self):
        assert_allclose(self.x_lege_1, data['x_lege_1'])

    def test_qnwlege_nodes_3d(self):
        assert_allclose(self.x_lege_3, data['x_lege_3'])

    def test_qnwlege_weights_1d(self):
        assert_allclose(self.w_lege_1, data['w_lege_1'])

    def test_qnwlege_weights_3d(self):
        assert_allclose(self.w_lege_3, data['w_lege_3'])


class TestQnwnorm(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.x_norm_1, cls.w_norm_1 = qnwnorm(n, a, b)
        cls.x_norm_3, cls.w_norm_3 = qnwnorm(n_3, mu_3d, sigma2_3d)

    def test_qnwnorm_nodes_1d(self):
        assert_allclose(self.x_norm_1, data['x_norm_1'])

    def test_qnwnorm_nodes_3d(self):
        assert_allclose(self.x_norm_3, data['x_norm_3'])

    def test_qnwnorm_weights_1d(self):
        assert_allclose(self.w_norm_1, data['w_norm_1'])

    def test_qnwnorm_weights_3d(self):
        assert_allclose(self.w_norm_3, data['w_norm_3'])


class TestQnwlogn(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.x_logn_1, cls.w_logn_1 = qnwlogn(n, a, b)
        cls.x_logn_3, cls.w_logn_3 = qnwlogn(n_3, mu_3d, sigma2_3d)

    def test_qnwlogn_nodes_1d(self):
        assert_allclose(self.x_logn_1, data['x_logn_1'])

    def test_qnwlogn_nodes_3d(self):
        assert_allclose(self.x_logn_3, data['x_logn_3'])

    def test_qnwlogn_weights_1d(self):
        assert_allclose(self.w_logn_1, data['w_logn_1'])

    def test_qnwlogn_weights_3d(self):
        assert_allclose(self.w_logn_3, data['w_logn_3'])


class TestQnwsimp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.x_simp_1, cls.w_simp_1 = qnwsimp(n, a, b)
        cls.x_simp_3, cls.w_simp_3 = qnwsimp(n_3, a_3, b_3)

    def test_qnwsimp_nodes_1d(self):
        assert_allclose(self.x_simp_1, data['x_simp_1'])

    def test_qnwsimp_nodes_3d(self):
        assert_allclose(self.x_simp_3, data['x_simp_3'])

    def test_qnwsimp_weights_1d(self):
        assert_allclose(self.w_simp_1, data['w_simp_1'])

    def test_qnwsimp_weights_3d(self):
        assert_allclose(self.w_simp_3, data['w_simp_3'])


class TestQnwtrap(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.x_trap_1, cls.w_trap_1 = qnwtrap(n, a, b)
        cls.x_trap_3, cls.w_trap_3 = qnwtrap(n_3, a_3, b_3)

    def test_qnwtrap_nodes_1d(self):
        assert_allclose(self.x_trap_1, data['x_trap_1'])

    def test_qnwtrap_nodes_3d(self):
        assert_allclose(self.x_trap_3, data['x_trap_3'])

    def test_qnwtrap_weights_1d(self):
        assert_allclose(self.w_trap_1, data['w_trap_1'])

    def test_qnwtrap_weights_3d(self):
        assert_allclose(self.w_trap_3, data['w_trap_3'])


class TestQnwunif(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.x_unif_1, cls.w_unif_1 = qnwunif(n, a, b)
        cls.x_unif_3, cls.w_unif_3 = qnwunif(n_3, a_3, b_3)

    def test_qnwunif_nodes_1d(self):
        assert_allclose(self.x_unif_1, data['x_unif_1'])

    def test_qnwunif_nodes_3d(self):
        assert_allclose(self.x_unif_3, data['x_unif_3'])

    def test_qnwunif_weights_1d(self):
        assert_allclose(self.w_unif_1, data['w_unif_1'])

    def test_qnwunif_weights_3d(self):
        assert_allclose(self.w_unif_3, data['w_unif_3'])


class TestQnwbeta(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.x_beta_1, cls.w_beta_1 = qnwbeta(n, b, b + 1.0)
        cls.x_beta_3, cls.w_beta_3 = qnwbeta(n_3, b_3, b_3 + 1.0)

    def test_qnwbeta_nodes_1d(self):
        assert_allclose(self.x_beta_1, data['x_beta_1'])

    def test_qnwbeta_nodes_3d(self):
        assert_allclose(self.x_beta_3, data['x_beta_3'])

    def test_qnwbeta_weights_1d(self):
        assert_allclose(self.w_beta_1, data['w_beta_1'])

    def test_qnwbeta_weights_3d(self):
        assert_allclose(self.w_beta_3, data['w_beta_3'])


class TestQnwgamm(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.x_gamm_1, cls.w_gamm_1 = qnwgamma(n, b)
        cls.x_gamm_3, cls.w_gamm_3 = qnwgamma(n_3, b_3)

    def test_qnwgamm_nodes_1d(self):
        assert_allclose(self.x_gamm_1, data['x_gamm_1'])

    def test_qnwgamm_nodes_3d(self):
        assert_allclose(self.x_gamm_3, data['x_gamm_3'])

    def test_qnwgamm_weights_1d(self):
        assert_allclose(self.w_gamm_1, data['w_gamm_1'])

    def test_qnwgamm_weights_3d(self):
        assert_allclose(self.w_gamm_3, data['w_gamm_3'])
