"""
Tests for vertex_enumeration.py

"""
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from nose.tools import eq_, raises
from quantecon.game_theory import NormalFormGame, vertex_enumeration
from quantecon.game_theory.vertex_enumeration import _BestResponsePolytope


class TestVertexEnumeration:
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
        bimatrix = [[(3, 3), (3, 3)],
                    [(2, 2), (5, 6)],
                    [(0, 3), (6, 1)]]
        d = {'g': NormalFormGame(bimatrix),
             'NEs': [([1, 0, 0], [1, 0]),
                     ([1, 0, 0], [2/3, 1/3]),
                     ([0, 1/3, 2/3], [1/3, 2/3])]}
        self.game_dicts.append(d)

    def test_vertex_enumeration(self):
        for d in self.game_dicts:
            NEs_computed = vertex_enumeration(d['g'])
            eq_(len(NEs_computed), len(d['NEs']))
            for NEs in (NEs_computed, d['NEs']):
                NEs.sort(key=lambda x: (list(x[0]), list(x[1])))
            for actions_computed, actions in zip(NEs_computed, d['NEs']):
                for action_computed, action in zip(actions_computed, actions):
                    assert_allclose(action_computed, action)


def test_vertex_enumeration_qhull_options():
    # Degenerate game, player 0's actions reordered
    bimatrix = [[(0, 3), (6, 1)],
                [(2, 2), (5, 6)],
                [(3, 3), (3, 3)]]
    g = NormalFormGame(bimatrix)
    NEs_expected = [([0, 0, 1], [1, 0]),
                    ([0, 0, 1], [2/3, 1/3]),
                    ([2/3, 1/3, 0], [1/3, 2/3])]
    qhull_options = 'QJ'
    NEs_computed = vertex_enumeration(g, qhull_options=qhull_options)
    eq_(len(NEs_computed), len(NEs_expected))
    for NEs in (NEs_computed, NEs_expected):
        NEs.sort(key=lambda x: (list(x[1]), list(x[0])))
    for actions_computed, actions in zip(NEs_computed, NEs_expected):
        for action_computed, action in zip(actions_computed, actions):
            assert_allclose(action_computed, action, atol=1e-10)


@raises(TypeError)
def test_vertex_enumeration_invalid_g():
    bimatrix = [[(3, 3), (3, 2)],
                [(2, 2), (5, 6)],
                [(0, 3), (6, 1)]]
    vertex_enumeration(bimatrix)


class TestBestResponsePolytope:
    def setUp(self):
        # From von Stengel 2007 in Algorithmic Game Theory
        bimatrix = [[(3, 3), (3, 2)],
                    [(2, 2), (5, 6)],
                    [(0, 3), (6, 1)]]
        g = NormalFormGame(bimatrix)

        # Original best reponse polytope for player 0
        vertices_P = np.array([
            [0, 0, 0],       # 0
            [1/3, 0, 0],     # a
            [2/7, 1/14, 0],  # b
            [0, 1/6, 0],     # c
            [0, 1/8, 1/4],   # d
            [0, 0, 1/3]      # e
        ])
        labelings_P = np.array([
            [0, 1, 2],  # 0
            [1, 2, 3],  # a
            [2, 3, 4],  # b
            [0, 2, 4],  # c
            [0, 3, 4],  # d
            [0, 1, 3]   # e
        ])

        # Sort rows lexicographically
        K = labelings_P.shape[1]
        ind = np.lexsort([labelings_P[:, K-k-1] for k in range(K)])
        self.labelings_P = labelings_P[ind]
        self.vertices_P = vertices_P[ind]

        # Translated best reponse polytope for player 0
        self.brp0 = _BestResponsePolytope(g.players[1], idx=0)

    def test_best_response_polytope(self):
        # Sort each row
        labelings_computed = np.sort(self.brp0.labelings, axis=1)

        # Sort rows lexicographically
        K = labelings_computed.shape[1]
        ind = np.lexsort([labelings_computed[:, K-k-1] for k in range(K)])
        labelings_computed = labelings_computed[ind]
        vertices_computed = \
            -self.brp0.equations[:, :-1] / self.brp0.equations[:, [-1]] + \
            1/self.brp0.trans_recip
        vertices_computed = vertices_computed[ind]

        assert_array_equal(labelings_computed, self.labelings_P)
        assert_allclose(vertices_computed, self.vertices_P, atol=1e-15)


@raises(TypeError)
def test_best_response_polytope_invalid_player_instance():
    bimatrix = [[(3, 3), (3, 2)],
                [(2, 2), (5, 6)],
                [(0, 3), (6, 1)]]
    g = NormalFormGame(bimatrix)
    _BestResponsePolytope(g)


if __name__ == '__main__':
    import sys
    import nose

    argv = sys.argv[:]
    argv.append('--verbose')
    argv.append('--nocapture')
    nose.main(argv=argv, defaultTest=__file__)
