"""
Filename: test_arma.py
Authors: Chase Coleman
Date: 07/24/2014

Tests for lqnash.py file.

"""
import sys
import os
import unittest
import numpy as np
from numpy.testing import assert_allclose
from quantecon.arma import ARMA


def test_noninteractive():
    "Test case for when agents don't affect each other"
    # Copied these values from test_lqcontrol
    a = eye(3)
    b1 = np.array([0, -1., 0.])
    b2 = np.array([0., 0., -1.])
    r1 = np.array([[25., -1., 0.], [-1., 2.5, 0], [0., 0., 0.]])
    r2 = np.array([[25., 0., -1.], [0., 0., 0.], [-1., 0., 2.5]])
    q1 = np.array([[-1.5]])
    q2 = np.array([[-1.5]])
    f1, f2, p1, p2 = nnash(a, b1, b2, -r1, -r2, -q1, -q2, 0, 0, 0, 0, 0, 0,
                           tol=1e-4, max_iter=10000)

