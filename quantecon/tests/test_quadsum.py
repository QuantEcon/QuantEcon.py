"""
Filename: test_tauchen.py
Authors: Chase Coleman
Date: 07/24/2014

Tests for quadsums.py file

"""
import sys
import os
import unittest
import numpy as np
from numpy.testing import assert_allclose
from quantecon.quadsums import var_quadratic_sum, m_quadratic_sum


def test_var_simplesum():
    beta = .95
    A = 1.
    C = 0.
    H = 1.
    x0 = 1.

    val = var_quadratic_sum(A, C, H, beta, x0)

    assert abs(val-20) < 1e-10


def test_var_identitysum():
    beta = .95
    A = np.eye(3)
    C = np.zeros((3, 3))
    H = np.eye(3)
    x0 = np.ones(3)

    val = var_quadratic_sum(A, C, H, beta, x0)

    assert abs(val-60) < 1e-10


def test_m_simplesum():
    pass


def test_m_identitysum():
    pass



if __name__ == '__main__':
    test_simplesum()
    test_identitysum()
    test_m_simplesum()
    test_m_identitysum