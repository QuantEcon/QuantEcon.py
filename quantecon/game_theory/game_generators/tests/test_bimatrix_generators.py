"""
Tests for bimatrix_generators.py

"""
import numpy as np
from numpy.testing import assert_array_equal

from quantecon.game_theory import sgc_game


def test_sgc_game():
    k = 2
    s = """\
        0.750 0.750 1.000 0.500 0.500 1.000 0.000 0.500 0.000 0.500 0.000 0.500
        0.000 0.500 0.500 1.000 0.750 0.750 1.000 0.500 0.000 0.500 0.000 0.500
        0.000 0.500 0.000 0.500 1.000 0.500 0.500 1.000 0.750 0.750 0.000 0.500
        0.000 0.500 0.000 0.500 0.000 0.500 0.500 0.000 0.500 0.000 0.500 0.000
        0.750 0.000 0.000 0.750 0.000 0.000 0.000 0.000 0.500 0.000 0.500 0.000
        0.500 0.000 0.000 0.750 0.750 0.000 0.000 0.000 0.000 0.000 0.500 0.000
        0.500 0.000 0.500 0.000 0.000 0.000 0.000 0.000 0.750 0.000 0.000 0.750
        0.500 0.000 0.500 0.000 0.500 0.000 0.000 0.000 0.000 0.000 0.000 0.750
        0.750 0.000"""
    bimatrix = np.fromstring(s, sep=' ')
    bimatrix.shape = (4*k-1, 4*k-1, 2)
    bimatrix = bimatrix.swapaxes(0, 1)

    g = sgc_game(k)
    assert_array_equal(g.payoff_profile_array, bimatrix)


if __name__ == '__main__':
    import sys
    import nose

    argv = sys.argv[:]
    argv.append('--verbose')
    argv.append('--nocapture')
    nose.main(argv=argv, defaultTest=__file__)
