"""
Tests for polymatrix_game.py
"""

from numpy.testing import assert_
from quantecon.game_theory.game_converters import qe_nfg_from_gam_file
from quantecon.game_theory.polymatrix_game import PolymatrixGame
from quantecon.game_theory import NormalFormGame
from itertools import product
from numpy import isclose


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
            if not isclose(
                nf1[action_combination][player],
                nf2[action_combination][player],
                atol=atol
            ):
                return False
    return True


def test_normal_form_to_polymatrix_to_normal_form():
    filename = "gam_files/big_polym.gam"
    nfg = qe_nfg_from_gam_file(filename)
    polym = PolymatrixGame.from_nf(nfg, is_polymatrix=True)
    back_in_nf = polym.to_nf()
    are_close = close_normal_form_games(nfg, back_in_nf)
    assert_(are_close)
