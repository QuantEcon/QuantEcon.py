"""
Filename: test_localint.py

Tests for localint.py

"""

import numpy as np
from numpy.testing import assert_array_equal, assert_equal

from quantecon.game_theory import LocalInteraction


class TestLocalInteraction:
    '''Test the methods of LocalInteraction'''

    def setUp(self):
        '''Setup a LocalInteraction instance'''
        payoff_matrix = np.asarray([[4, 0], [2, 3]])
        adj_matrix = np.asarray([[0, 1, 3],
                                 [2, 0, 1],
                                 [3, 2, 0]])
        self.li = LocalInteraction(payoff_matrix, adj_matrix)

    def test_play(self):
        init_actions = (0, 0, 1)
        x = (1, 0, 0)
        assert_equal(self.li.play(init_actions=init_actions), x)

    def test_time_series_simultaneous_revision(self):
        init_actions = (0, 0, 1)
        x = [[0, 0, 1],
             [1, 0, 0],
             [0, 1, 1]]
        assert_array_equal(self.li.time_series(ts_length=3,
                                               init_actions=init_actions), x)

    def test_time_series_asynchronous_revision(self):
        seed = 1234
        init_actions = (0, 0, 1)
        x = [self.li.time_series(ts_length=3,
                                 revision='asynchronous',
                                 init_actions=init_actions,
                                 random_state=np.random.RandomState(seed))
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
                                               init_actions=init_actions,
                                               player_ind_seq=player_ind_seq),
                           x)


if __name__ == '__main__':
    import sys
    import nose

    argv = sys.argv[:]
    argv.append('--verbose')
    argv.append('--nocapture')
    nose.main(argv=argv, defaultTest=__file__)
