"""
Tests for polymatrix_game.py
"""

from numpy.testing import assert_, assert_raises
from quantecon.game_theory.game_converters import qe_nfg_from_gam_file
from quantecon.game_theory.polymatrix_game import PolymatrixGame
from quantecon.game_theory import NormalFormGame
from numpy import allclose

import os

# Mimicing quantecon.tests.util.get_data_dir
data_dir_name = "gam_files"
this_dir = os.path.dirname(__file__)
data_dir = os.path.join(this_dir, data_dir_name)


def close_normal_form_games(
        nf1: NormalFormGame,
        nf2: NormalFormGame,
        atol=1e-4
) -> bool:
    if nf1.N != nf2.N:
        return False
    for player in range(nf1.N):
        if nf1.nums_actions[player] != nf2.nums_actions[player]:
            return False
    for player in range(nf1.N):
        if not allclose(
            nf1.players[player].payoff_array,
            nf2.players[player].payoff_array,
            atol=atol
        ):
            return False
    return True


class TestPolymatrixGame():
    @classmethod
    def setup_class(cls):
        filename = "minimum_effort_game.gam"
        cls.non_pmg = qe_nfg_from_gam_file(
            os.path.join(data_dir, filename))
        filename = "big_polym.gam"
        cls.pmg1 = qe_nfg_from_gam_file(
            os.path.join(data_dir, filename))
        filename = "triggers_back_case.gam"
        cls.pmg2 = qe_nfg_from_gam_file(
            os.path.join(data_dir, filename))
        bimatrix = [[(54, 23), (72, 34)],
                    [(92, 32), (34, 36)],
                    [(57, 54), (76, 85)]]
        cls.bimatrix_game = NormalFormGame(bimatrix)

    def test_different_games_are_not_close(self):
        are_close = close_normal_form_games(self.pmg1, self.pmg2)
        assert_(not are_close)

    def test_same_games_are_close(self):
        are_close = close_normal_form_games(self.pmg1, self.pmg1)
        assert_(are_close)

    def test_normal_form_to_polymatrix_to_normal_form_multiplayer(self):
        polymg = PolymatrixGame.from_nf(self.pmg1, is_polymatrix=True)
        back_in_nf = polymg.to_nfg()
        are_close = close_normal_form_games(self.pmg1, back_in_nf)
        assert_(are_close)

    def test_normal_form_to_polymatrix_to_normal_form_bimatrix(self):
        polymg = PolymatrixGame.from_nf(
            self.bimatrix_game, is_polymatrix=True)
        back_in_nf = polymg.to_nfg()
        are_close = close_normal_form_games(
            self.bimatrix_game, back_in_nf)
        assert_(are_close)

    def test_comes_up_with_approximation(self):
        polymg = PolymatrixGame.from_nf(
            self.non_pmg, is_polymatrix=False)
        back_in_nf = polymg.to_nfg()
        assert_(not close_normal_form_games(
            self.non_pmg,
            back_in_nf
        ))

    def test_nonpolymatrix_gets_error_if_is_polymatrix_flag(self):
        with assert_raises(AssertionError):
            PolymatrixGame.from_nf(
                self.non_pmg,
                is_polymatrix=True
            )
