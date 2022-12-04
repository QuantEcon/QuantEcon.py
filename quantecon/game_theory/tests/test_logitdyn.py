"""
Filename: test_logitdyn.py

Tests for logitdyn.py

"""
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal_nulp

from quantecon.game_theory import NormalFormGame, LogitDynamics


class TestLogitDynamics:
    '''Test the methods of LogitDynamics'''

    def setup_method(self):
        '''Setup a LogitDynamics instance'''
        # symmetric 2x2 coordination game
        payoff_matrix = [[4, 0],
                         [3, 2]]
        beta = 4.0
        data = NormalFormGame(payoff_matrix)
        self.ld = LogitDynamics(data, beta=beta)

    def test_play(self):
        seed = 76240103339929371127372784081282227092
        x = [self.ld.play(init_actions=(0, 0),
                          random_state=np.random.default_rng(seed))
             for i in range(2)]
        assert_array_equal(x[0], x[1])

    def test_time_series(self):
        seed = 165570719993771384215214311194249493239
        series = [self.ld.time_series(ts_length=10, init_actions=(0, 0),
                                      random_state=np.random.default_rng(seed))
                  for i in range(2)]
        assert_array_equal(series[0], series[1])


def test_set_choice_probs_with_asymmetric_payoff_matrix():
    bimatrix = np.array([[(4, 4), (1, 1), (0, 3)],
                         [(3, 0), (1, 1), (2, 2)]])
    beta = 1.0
    data = NormalFormGame(bimatrix)
    ld = LogitDynamics(data, beta=beta)

    # (Normalized) CDFs of logit choice
    cdfs = np.ones((bimatrix.shape[1], bimatrix.shape[0]))
    cdfs[:, 0] = 1 / (1 + np.exp(beta*(bimatrix[1, :, 0]-bimatrix[0, :, 0])))

    # self.ld.players[0].logit_choice_cdfs: unnormalized
    cdfs_computed = ld.players[0].logit_choice_cdfs
    cdfs_computed = cdfs_computed / cdfs_computed[..., [-1]]  # Normalized

    assert_array_almost_equal_nulp(cdfs_computed, cdfs)
