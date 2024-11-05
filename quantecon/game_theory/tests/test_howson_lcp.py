"""
Tests for howson_lcp.py
"""

from numpy.testing import assert_
from quantecon.game_theory.game_converters import qe_nfg_from_gam_file
from quantecon.game_theory.howson_lcp import polym_lcp_solver
from quantecon.game_theory.polymatrix_game import PolymatrixGame


def test_polym_lcp_solver_where_solution_is_pure_NE():
    filename = "gam_files/big_polym.gam"
    nfg = qe_nfg_from_gam_file(filename)
    polym = PolymatrixGame.from_nf(nfg)
    ne = polym_lcp_solver(polym)
    worked = nfg.is_nash(ne)
    assert_(worked)

def test_polym_lcp_solver_where_lcp_solver_must_backtrack():
    filename = "gam_files/triggers_back_case.gam"
    nfg = qe_nfg_from_gam_file(filename)
    polym = PolymatrixGame.from_nf(nfg)
    ne = polym_lcp_solver(polym)
    worked = nfg.is_nash(ne)
    assert_(worked)
