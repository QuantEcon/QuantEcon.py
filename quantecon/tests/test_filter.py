"""
Tests for filter.py.
Using the data of original paper.

"""

import os
import pandas as pd
import numpy as np
from numpy.testing import assert_allclose
from quantecon.filter import hamilton_filter
from quantecon.tests.util import get_data_dir

def test_hamilton_filter():
    # read data
    data_dir = get_data_dir()
    data = pd.read_csv(os.path.join(data_dir, "employment.csv"),
                       names = ['year', 'employment', 'matlab_cyc', 'matlab_cyc_rw'])

    data['hamilton_cyc'], data['hamilton_trend'] =  hamilton_filter(
        100*np.log(data['employment']), 8, 4)
    data['hamilton_cyc_rw'], data['hamilton_trend_rw'] = hamilton_filter(
        100*np.log(data['employment']), 8)
    assert_allclose(data['matlab_cyc'], data['hamilton_cyc'],
                    rtol = 1e-07, atol = 1e-07)
    assert_allclose(data['matlab_cyc_rw'], data['hamilton_cyc_rw'])


if __name__ == '__main__':
    test_hamilton_filter()
