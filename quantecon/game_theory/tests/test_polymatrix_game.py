"""
Tests for polymatrix_game.py
"""

from numpy.testing import assert_, assert_raises
from quantecon.game_theory.game_converters import qe_nfg_from_gam_file
from quantecon.game_theory.polymatrix_game import PolymatrixGame
from quantecon.game_theory import NormalFormGame
from itertools import product
from numpy import isclose

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
    for action_combination in product(*[
            range(a) for a in nf1.nums_actions]):
        for player in range(nf1.N):
            # Will change this to use player's payoff functions
            # for efficiency
            if not isclose(
                nf1[action_combination][player],
                nf2[action_combination][player],
                atol=atol
            ):
                return False
    return True


def test_different_games_are_not_close():
    filename = "big_polym.gam"
    nfg1 = qe_nfg_from_gam_file(os.path.join(data_dir, filename))
    filename = "triggers_back_case.gam"
    nfg2 = qe_nfg_from_gam_file(os.path.join(data_dir, filename))
    are_close = close_normal_form_games(nfg1, nfg2)
    assert_(not are_close)


def test_normal_form_to_polymatrix_to_normal_form_multiplayer():
    filename = "big_polym.gam"
    nfg = qe_nfg_from_gam_file(os.path.join(data_dir, filename))
    polymg = PolymatrixGame.from_nf(nfg, is_polymatrix=True)
    back_in_nf = polymg.to_nfg()
    are_close = close_normal_form_games(nfg, back_in_nf)
    assert_(are_close)


def test_normal_form_to_polymatrix_to_normal_form_bimatrix():
    bimatrix = [[(54, 23), (72, 34)],
                [(92, 32), (34, 36)],
                [(57, 54), (76, 85)]]
    nfg = NormalFormGame(bimatrix)
    polymg = PolymatrixGame.from_nf(nfg, is_polymatrix=True)
    back_in_nf = polymg.to_nfg()
    are_close = close_normal_form_games(nfg, back_in_nf)
    assert_(are_close)


class TestApproximatingPolymatrix():
    def setup_method(self):
        filename = "minimum_effort_game.gam"
        self.nfg = qe_nfg_from_gam_file(
            os.path.join(data_dir, filename))

    def test_comes_up_with_approximation(self):
        polymg = PolymatrixGame.from_nf(
            self.nfg, is_polymatrix=False)
        back_in_nf = polymg.to_nfg()
        assert_(not close_normal_form_games(
            self.nfg,
            back_in_nf
        ))

    def test_nonpolymatrix_gets_error_if_is_polymatrix_flag(self):
        with assert_raises(AssertionError):
            PolymatrixGame.from_nf(
                self.nfg,
                is_polymatrix=True
            )
