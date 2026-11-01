"""
Filename: test_localint.py

Tests for localint.py

"""
import numpy as np
from numpy.testing import assert_array_equal, assert_equal, assert_
import pytest
from scipy import sparse

from quantecon.game_theory import LocalInteraction


# Both scipy sparse families: the old "matrix" family and the new
# "array" family
sparse_families = [sparse.csr_matrix, sparse.csr_array]


class TestLocalInteraction:
    '''Test the methods of LocalInteraction'''

    @pytest.fixture(autouse=True, params=[np.asarray] + sparse_families,
                    ids=['dense', 'csr_matrix', 'csr_array'])
    def setup(self, request):
        '''Setup a LocalInteraction instance with dense or sparse input'''
        payoff_matrix = np.asarray([[4, 0], [2, 3]])
        adj_matrix = request.param([[0, 1, 3],
                                    [2, 0, 1],
                                    [3, 2, 0]])
        self.li = LocalInteraction(payoff_matrix, adj_matrix)

    def test_play(self):
        init_actions = (0, 0, 1)
        x = (1, 0, 0)
        assert_equal(self.li.play(actions=init_actions), x)

    def test_time_series_simultaneous_revision(self):
        init_actions = (0, 0, 1)
        x = [[0, 0, 1],
             [1, 0, 0],
             [0, 1, 1]]
        assert_array_equal(self.li.time_series(ts_length=3,
                                               actions=init_actions), x)

    def test_time_series_asynchronous_revision(self):
        seed = 21388225457580135037104913043871211454
        init_actions = (0, 0, 1)
        x = [self.li.time_series(ts_length=3,
                                 revision='asynchronous',
                                 actions=init_actions,
                                 random_state=np.random.default_rng(seed))
             for i in range(2)]
        assert_array_equal(x[0], x[1])

    def test_time_series_asynchronous_revision_with_player_index(self):
        init_actions = (0, 0, 1)
        player_ind_seq = [0, 1, 2]
        x = [[0, 0, 1],
             [1, 0, 1],
             [1, 1, 1]]
        assert_array_equal(self.li.time_series(ts_length=3,
                                               revision='asynchronous',
                                               actions=init_actions,
                                               player_ind_seq=player_ind_seq),
                           x)


def test_adj_matrix_dense_input():
    # Dense input gives a csr_matrix (old family)
    payoff_matrix = [[4, 0], [2, 3]]
    adj_matrix = np.array([[0, 1, 3], [2, 0, 1], [3, 2, 0]])
    li = LocalInteraction(payoff_matrix, adj_matrix)
    assert_(type(li.adj_matrix) is sparse.csr_matrix)
    assert_array_equal(li.adj_matrix.toarray(), adj_matrix)


def test_adj_matrix_sparse_input_family_preserved():
    payoff_matrix = [[4, 0], [2, 3]]
    adj_matrix = np.array([[0, 1, 3], [2, 0, 1], [3, 2, 0]])
    for sparse_type in sparse_families:
        # CSR input, and non-CSR input converted to CSR in the same family
        for A in [sparse_type(adj_matrix), sparse_type(adj_matrix).tocoo()]:
            li = LocalInteraction(payoff_matrix, A)
            assert_(type(li.adj_matrix) is sparse_type)
            assert_array_equal(li.adj_matrix.toarray(), adj_matrix)
