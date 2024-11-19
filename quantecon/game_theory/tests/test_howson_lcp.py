"""
Tests for howson_lcp.py
"""

import numpy as np
from numpy.testing import assert_
from quantecon.game_theory.game_converters import qe_nfg_from_gam_file
from quantecon.game_theory.howson_lcp import polym_lcp_solver
from quantecon.game_theory.polymatrix_game import PolymatrixGame

import os


# Mimicing quantecon.tests.util.get_data_dir
data_dir_name = "gam_files"
this_dir = os.path.dirname(__file__)
data_dir = os.path.join(this_dir, data_dir_name)


def test_polym_lcp_solver_where_solution_is_pure_NE():
    filename = "big_polym.gam"
    nfg = qe_nfg_from_gam_file(os.path.join(data_dir, filename))
    polymg = PolymatrixGame.from_nf(nfg)
    ne = polym_lcp_solver(polymg)
    worked = nfg.is_nash(ne)
    assert_(worked)


def test_polym_lcp_solver_where_lcp_solver_must_backtrack():
    filename = "triggers_back_case.gam"
    nfg = qe_nfg_from_gam_file(os.path.join(data_dir, filename))
    polymg = PolymatrixGame.from_nf(nfg)
    ne = polym_lcp_solver(polymg)
    worked = nfg.is_nash(ne)
    assert_(worked)


def test_solves_rock_paper_scissors():
    polymatrix = {
        (0, 1): np.array([
            [0, -1,  1],
            [1,  0, -1],
            [-1,  1,  0]
        ]),
        (1, 0): np.array([
            [0, -1,  1],
            [1,  0, -1],
            [-1,  1,  0]
        ])
    }
    polymg = PolymatrixGame(
        2,
        [3, 3],
        polymatrix
    )
    nfg = polymg.to_nf()
    ne = polym_lcp_solver(polymg)
    worked = nfg.is_nash(ne)
    print(ne)
    assert_(worked)


def test_solves_rps_with_scissorless_third_player():
    """
    Rock Paper Scissors, with an extra player, who
    just has rock and paper.
    """
    polymatrix = {
        (0, 1): np.array([
            [  0, -10,  10],
            [ 10,   0, -10],
            [-10,  10,   0]
        ]),
        (0, 2): np.array([
            [  0, -10],
            [ 15,   0],
            [-10,   0]
        ]),
        (1, 0): np.array([
            [  0, -10,  10],
            [ 10,   0, -10],
            [-10,  10,   0]
        ]),
        (1, 2): np.array([
            [  0, -10],
            [ 15,   0],
            [-10,   0]
        ]),
        (2, 0): np.array([
            [ 0, -10,  10],
            [10,   0, -10],
        ]),
        (2, 1): np.array([
            [ 0, -10,  10],
            [10,   0, -10],
        ])
    }
    polymg = PolymatrixGame(
        3,
        [3, 3, 2],
        polymatrix
    )
    nfg = polymg.to_nf()
    ne = polym_lcp_solver(polymg)
    worked = nfg.is_nash(ne)
    print(ne)
    assert_(worked)


def test_solves_rps_with_rocking_third_player():
    """
    Rock Paper Scissors, with an extra player, who
    has a rock, paper, and bigger rock.
    """
    polymatrix = {
        (0, 1): np.array([
            [  0, -10,  10],
            [ 10,   0, -10],
            [-10,  10,   0]
        ]),
        (0, 2): np.array([
            [  0, -10,   0],
            [ 15,   0, -10],
            [-10,   0, -25]
        ]),
        (1, 0): np.array([
            [  0, -10,  10],
            [ 10,   0, -10],
            [-10,  10,   0]
        ]),
        (1, 2): np.array([
            [  0, -10,   0],
            [ 15,   0, -10],
            [-10,   0, -25]
        ]),
        (2, 0): np.array([
            [ 0, -10,  10],
            [10,   0, -10],
            [ 0, -15,  25]
        ]),
        (2, 1): np.array([
            [ 0, -10,  10],
            [10,   0, -10],
            [ 0, -15,  25]
        ])
    }
    polymg = PolymatrixGame(
        3,
        [3, 3, 3],
        polymatrix
    )
    nfg = polymg.to_nf()
    ne = polym_lcp_solver(polymg)
    worked = nfg.is_nash(ne)
    print(ne)
    assert_(worked)
