"""
Author: Daisuke Oyama

Tests for support_enumeration.py

"""
from numpy.testing import assert_allclose
from quantecon.game_theory import Player, NormalFormGame, support_enumeration


class TestSupportEnumeration():
    def setUp(self):
        self.game_dicts = []

        # From von Stengel 2007 in Algorithmic Game Theory
        bimatrix = [[(3, 3), (3, 2)],
                    [(2, 2), (5, 6)],
                    [(0, 3), (6, 1)]]
        d = {'g': NormalFormGame(bimatrix),
             'NEs': [([1, 0, 0], [1, 0]),
                     ([4/5, 1/5, 0], [2/3, 1/3]),
                     ([0, 1/3, 2/3], [1/3, 2/3])]}
        self.game_dicts.append(d)

        # Degenerate game
        # NEs ([0, p, 1-p], [1/2, 1/2]), 0 <= p <= 1, are not detected.
        bimatrix = [[(1, 1), (-1, 0)],
                    [(-1, 0), (1, 0)],
                    [(0, 0), (0, 0)]]
        d = {'g': NormalFormGame(bimatrix),
             'NEs': [([1, 0, 0], [1, 0]),
                     ([0, 1, 0], [0, 1])]}
        self.game_dicts.append(d)

    def test_support_enumeration(self):
        for d in self.game_dicts:
            NEs_computed = support_enumeration(d['g'])
            for actions_computed, actions in zip(NEs_computed, d['NEs']):
                for action_computed, action in zip(actions_computed, actions):
                    assert_allclose(action_computed, action)


if __name__ == '__main__':
    import sys
    import nose

    argv = sys.argv[:]
    argv.append('--verbose')
    argv.append('--nocapture')
    nose.main(argv=argv, defaultTest=__file__)
