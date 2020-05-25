"""
Tests for repeated_game.py

"""
import numpy as np
from numpy.testing import assert_allclose
from quantecon.game_theory import NormalFormGame, RepeatedGame


class TestAS():
    def setUp(self):
        self.game_dicts = []

        # Example from Abreu and Sannikov (2014)
        bimatrix = [[(16, 9), (3, 13), (0, 3)],
                    [(21, 1), (10, 4), (-1, 0)],
                    [(9, 0), (5, -4), (-5, -15)]]
        vertices = np.array([[7.33770472e+00, 1.09826253e+01],
                             [1.12568240e+00, 2.80000000e+00],
                             [7.33770472e+00, 4.44089210e-16],
                             [7.86308964e+00, 4.44089210e-16],
                             [1.97917977e+01, 2.80000000e+00],
                             [1.55630896e+01, 9.10000000e+00]])
        d = {'sg': NormalFormGame(bimatrix),
             'delta': 0.3,
             'vertices': vertices,
             'u': np.zeros(2)}
        self.game_dicts.append(d)

        # Prisoner's dilemma
        bimatrix = [[(9, 9), (1, 10)],
                    [(10, 1), (3, 3)]]
        vertices = np.array([[3.  , 3.  ],
                             [9.75, 3.  ],
                             [9.  , 9.  ],
                             [3.  , 9.75]])
        d = {'sg': NormalFormGame(bimatrix),
             'delta': 0.9,
             'vertices': vertices,
             'u': np.array([3., 3.])}
        self.game_dicts.append(d)

    def test_abreu_sannikov(self):
        for d in self.game_dicts:
            rpg = RepeatedGame(d['sg'], d['delta'])
            for method in ('abreu_sannikov', 'AS'):
                hull = rpg.equilibrium_payoffs(method=method,
                                               options={'u_init': d['u']})
                assert_allclose(hull.points[hull.vertices], d['vertices'])

if __name__ == '__main__':
    import sys
    import nose

    argv = sys.argv[:]
    argv.append('--verbose')
    argv.append('--nocapture')
    nose.main(argv=argv, defaultTest=__file__)
