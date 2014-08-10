"""
tests for quantecon.util

"""
from __future__ import division
from collections import Counter
import unittest
import numpy as np
from numpy.testing import assert_allclose
from nose.plugins.attrib import attr
import pandas as pd
from quantecon import util as qeu


def test_solve_discrete_lyapunov_zero():
    'Simple test where X is all zeros'
    A = np.eye(4) * .95
    B = np.zeros((4, 4))

    X = qeu.solve_discrete_lyapunov(A, B)

    assert_allclose(X, np.zeros((4, 4)))


def test_solve_discrete_lyapunov_one():
    'Simple test where X is all ones'
