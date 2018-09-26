"""
Tests for repeated_game.py

"""
import numpy as np
from numpy.testing import assert_array_equal
from quantecon.game_theory import Player, NormalFormGame, RepeatedGame

class TestAS():
    def setUp(self):
        self.game_dicts = []

        # Example from Abreu and Sannikov (2014)
        bimatrix = [[(16, 9), (3, 13), (0, 3)],
                    [(21, 1), (10, 4), (-1, 0)],
                    [(9, 0), (5, -4), (-5, -15)]]
        vertice_inds = np.array([4, 8, 12, 11, 6, 1])
        d = {'sg': NormalFormGame(bimatrix),
             'delta': 0.3,
             'vertices': vertice_inds,
             'u': np.array([0, 0])}
        self.game_dicts.append(d)

        # Prisoner's dilemma
        bimatrix = [[(9, 9), (1, 10)],
                    [(10, 1), (3, 3)]]
        vertice_inds = np.array([6, 3, 0, 1])
        d = {'sg': NormalFormGame(bimatrix),
             'delta': 0.9,
             'vertices': vertice_inds,
             'u': np.array([3, 3])}
        self.game_dicts.append(d)

    def test_AS(self):
        for d in self.game_dicts:
            rpg = RepeatedGame(d['sg'], d['delta'])
            hull = rpg.AS(u=d['u'])
            assert_array_equal(d['vertices'], hull.vertices)

if __name__ == '__main__':
    import sys
    import nose

    argv = sys.argv[:]
    argv.append('--verbose')
    argv.append('--nocapture')
    nose.main(argv=argv, defaultTest=__file__)
