"""
Tests for bimatrix_generators.py

"""
import numpy as np
from numpy.testing import assert_array_equal
from nose.tools import eq_, ok_
from quantecon.gridtools import num_compositions

from quantecon.game_theory import blotto_game, sgc_game


class TestBlottoGame:
    def setUp(self):
        self.h, self.t = 4, 3
        rho = 0.5
        self.g = blotto_game(self.h, self.t, rho)

    def test_size(self):
        n = num_compositions(self.h, self.t)
        eq_(self.g.nums_actions, (n, n))

    def test_constant_diagonal(self):
        for i in range(self.g.N):
            diag = self.g.payoff_arrays[i].diagonal()
            ok_((diag == diag[0]).all())

    def test_seed(self):
        seed = 0
        h, t = 3, 4
        rho = -0.5
        g0 = blotto_game(h, t, rho, random_state=seed)
        g1 = blotto_game(h, t, rho, random_state=seed)
        assert_array_equal(g1.payoff_profile_array, g0.payoff_profile_array)


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
