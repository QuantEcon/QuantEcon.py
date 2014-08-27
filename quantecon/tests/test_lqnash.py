"""
Filename: test_arma.py
Authors: Chase Coleman
Date: 07/24/2014

Tests for lqnash.py file.

"""
from __future__ import division
import sys
import os
import unittest
import numpy as np
from numpy.testing import assert_allclose
from quantecon.lqnash import nnash
from quantecon.lqcontrol import LQ


def test_noninteractive():
    "Test case for when agents don't interact with each other"
    # Copied these values from test_lqcontrol
    a = np.array([[.95, 0.], [0, .95]])
    b1 = np.array([.95, 0.])
    b2 = np.array([0., .95])
    r1 = np.array([[-.25, 0.], [0., 0.]])
    r2 = np.array([[0., 0.], [0., -.25]])
    q1 = np.array([[-.15]])
    q2 = np.array([[-.15]])
    f1, f2, p1, p2 = nnash(a, b1, b2, r1, r2, q1, q2, 0, 0, 0, 0, 0, 0,
                           tol=1e-8, max_iter=10000)

    alq = a[:1, :1]
    blq = b1[:1].reshape((1, 1))
    rlq = r1[:1, :1]
    qlq = q1

    lq_obj = LQ(qlq, rlq, alq, blq, beta=1.)
    p, f, d = lq_obj.stationary_values()

    assert_allclose(f1, f2[:, ::-1])
    assert_allclose(f1[0, 0], f[0])
    assert_allclose(p1[0, 0], p2[1, 1])
    assert_allclose(p1[0, 0], p[0, 0])

