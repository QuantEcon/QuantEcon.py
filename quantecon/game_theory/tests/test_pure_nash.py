"""
Tests for pure_nash.py

"""

import numpy as np
import itertools
from nose.tools import eq_
from quantecon.game_theory import NormalFormGame, pure_nash_brute


class TestPureNashBruteForce():
    def setUp(self):
        self.game_dicts = []

        # Matching Pennies game with no pure nash equilibrium
        MP_bimatrix = [[(1, -1), (-1, 1)],
                       [(-1, 1), (1, -1)]]
        MP_NE = []

        # Prisoners' Dilemma game with one pure nash equilibrium
        PD_bimatrix = [[(1, 1), (-2, 3)],
                       [(3, -2), (0, 0)]]
        PD_NE = [(1, 1)]

        # Battle of the Sexes game with two pure nash equilibria
        BoS_bimatrix = [[(3, 2), (1, 0)],
                        [(0, 1), (2, 3)]]
        BoS_NE = [(0, 0), (1, 1)]

        # Unanimity Game with more than two players
        N = 4
        a, b = 1, 2
        g_Unanimity = NormalFormGame((2,)*N)
        g_Unanimity[(0,)*N] = (a,)*N
        g_Unanimity[(1,)*N] = (b,)*N

        Unanimity_NE = [(0,)*N]
        for k in range(2, N-2+1):
            for ind in itertools.combinations(range(N), k):
                a = np.ones(N, dtype=int)
                a[list(ind)] = 0
                Unanimity_NE.append(tuple(a))
        Unanimity_NE.append((1,)*N)

        for bimatrix, NE in zip([MP_bimatrix, PD_bimatrix, BoS_bimatrix],
                                [MP_NE, PD_NE, BoS_NE]):
            d = {'g': NormalFormGame(bimatrix),
                 'NEs': NE}
            self.game_dicts.append(d)
        self.game_dicts.append({'g': g_Unanimity,
                                'NEs': Unanimity_NE})

    def test_brute_force(self):
        for d in self.game_dicts:
            eq_(sorted(pure_nash_brute(d['g'])), sorted(d['NEs']))

    def test_tol(self):
        # Prisoners' Dilemma game with one NE and one epsilon NE
        epsilon = 1e-08

        PD_bimatrix = [[(1, 1), (-2, 1 + epsilon)],
                       [(1 + epsilon, -2), (0, 0)]]

        NEs = [(1, 1)]
        epsilon_NEs = [(1, 1), (0, 0)]

        g = NormalFormGame(PD_bimatrix)
        for tol, answer in zip([0, epsilon], [NEs, epsilon_NEs]):
            eq_(sorted(pure_nash_brute(g, tol=tol)), sorted(answer))


if __name__ == '__main__':
    import sys
    import nose

    argv = sys.argv[:]
    argv.append('--verbose')
    argv.append('--nocapture')
    nose.main(argv=argv, defaultTest=__file__)
