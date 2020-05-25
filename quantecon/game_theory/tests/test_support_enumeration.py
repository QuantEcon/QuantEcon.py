"""
Tests for support_enumeration.py

"""
import numpy as np
from numpy.testing import assert_allclose
from nose.tools import eq_, raises
from quantecon.util import check_random_state
from quantecon.game_theory import Player, NormalFormGame, support_enumeration


def random_skew_sym(n, m=None, random_state=None):
    """
    Generate a random skew symmetric zero-sum NormalFormGame of the form
    O    B
    -B.T O
    where B is an n x m matrix.

    """
    if m is None:
        m = n
    random_state = check_random_state(random_state)
    B = random_state.random_sample((n, m))
    A = np.empty((n+m, n+m))
    A[:n, :n] = 0
    A[n:, n:] = 0
    A[:n, n:] = B
    A[n:, :n] = -B.T
    return NormalFormGame([Player(A) for i in range(2)])


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
            eq_(len(NEs_computed), len(d['NEs']))
            for actions_computed, actions in zip(NEs_computed, d['NEs']):
                for action_computed, action in zip(actions_computed, actions):
                    assert_allclose(action_computed, action)

    def test_no_error_skew_sym(self):
        # Test no LinAlgError is raised.
        n, m = 3, 2
        seed = 7028
        g = random_skew_sym(n, m, random_state=seed)
        support_enumeration(g)


@raises(TypeError)
def test_support_enumeration_invalid_g():
    bimatrix = [[(3, 3), (3, 2)],
                [(2, 2), (5, 6)],
                [(0, 3), (6, 1)]]
    support_enumeration(bimatrix)


if __name__ == '__main__':
    import sys
    import nose

    argv = sys.argv[:]
    argv.append('--verbose')
    argv.append('--nocapture')
    nose.main(argv=argv, defaultTest=__file__)
