"""
Filename: test_filter.py
Authors: Shunsuke Hori

Tests for filter.py.
Using the data of original paper.

"""

import sys
import os
import unittest
import pandas as pd
import numpy as np
from numpy.testing import assert_allclose
from quantecon.filter import hamilton_filter
from quantecon.tests.util import get_data_dir

def test_hamilton_filter():
    # read data
    data_dir = get_data_dir()
    data = pd.read_csv(os.path.join(data_dir, "employment.csv"),
                       names = ['year', 'employment', 'matlab_cycle'])

    filtered_data = hamilton_filter(100*np.log(data['employment']), 8, 4, 'empl_')
    assert_allclose(data['matlab_cycle'], filtered_data['empl_cycle'],
                    rtol = 1e-07, atol = 1e-07)


if __name__ == '__main__':
    test_hamilton_filter()
