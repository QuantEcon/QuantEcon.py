"""
Tests for minmax

"""
import numpy as np
from numpy.testing import assert_, assert_allclose

from quantecon.optimize import minmax
from quantecon.game_theory import NormalFormGame, Player, lemke_howson


class TestMinmax:
    def test_RPS(self):
        A = np.array(
            [[0, 1, -1],
             [-1, 0, 1],
             [1, -1, 0],
             [-1, -1, -1]]
        )
        v_expected = 0
        x_expected = [1/3, 1/3, 1/3, 0]
        y_expected = [1/3, 1/3, 1/3]

        v, x, y = minmax(A)

        assert_allclose(v, v_expected)
        assert_allclose(x, x_expected)
        assert_allclose(y, y_expected)

    def test_random_matrix(self):
        seed = 12345
        rng = np.random.default_rng(seed)
        size = (10, 15)
        A = rng.normal(size=size)
        v, x, y = minmax(A)

        for z in [x, y]:
            assert_((z >= 0).all())
            assert_allclose(z.sum(), 1)

        g = NormalFormGame((Player(A), Player(-A.T)))
        NE = lemke_howson(g)
        assert_allclose(v, NE[0] @ A @ NE[1])
        assert_(g.is_nash((x, y)))
