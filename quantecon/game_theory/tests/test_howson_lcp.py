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
    assert_(worked)


def test_solves_rps_with_scissorless_third_player():
    """
    Rock Paper Scissors, with an extra player, who
    just has rock and paper.
    """
    polymatrix = {
        (0, 1): np.array([
            [0, -10,  10],
            [10,   0, -10],
            [-10,  10,   0]
        ]),
        (0, 2): np.array([
            [0, -10],
            [15,   0],
            [-10,   0]
        ]),
        (1, 0): np.array([
            [0, -10,  10],
            [10,   0, -10],
            [-10,  10,   0]
        ]),
        (1, 2): np.array([
            [0, -10],
            [15,   0],
            [-10,   0]
        ]),
        (2, 0): np.array([
            [0, -10,  10],
            [10,   0, -10],
        ]),
        (2, 1): np.array([
            [0, -10,  10],
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
    assert_(worked)


def test_solves_rps_with_rocking_third_player():
    """
    Rock Paper Scissors, with an extra player, who
    has a rock, paper, and bigger rock.
    """
    polymatrix = {
        (0, 1): np.array([
            [0, -10,  10],
            [10,   0, -10],
            [-10,  10,   0]
        ]),
        (0, 2): np.array([
            [0, -10,   0],
            [15,   0, -10],
            [-10,   0, -25]
        ]),
        (1, 0): np.array([
            [0, -10,  10],
            [10,   0, -10],
            [-10,  10,   0]
        ]),
        (1, 2): np.array([
            [0, -10,   0],
            [15,   0, -10],
            [-10,   0, -25]
        ]),
        (2, 0): np.array([
            [0, -10,  10],
            [10,   0, -10],
            [0, -15,  25]
        ]),
        (2, 1): np.array([
            [0, -10,  10],
            [10,   0, -10],
            [0, -15,  25]
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
    assert_(worked)


def test_solves_multiplayer_rps_like():
    """
    The players and their actions are listed.
    Schoolboy: Rock, Paper, Scissors.
    Farmer: Chicken, Bull.
    Zookeeper: Parrott, Monkey.
    Curator: Mummy, Painting.
    Archeologist: Mammoth, Pterradon, Tyrannosaurus.
    Entomologist: Mosquito, Ant.
    (The reasons for the payoffs can be found with imagination.)
    """
    polymatrix = {
        (0, 1): np.array([
            [2, 0],
            [-5, 5],
            [7, 0]
        ]),
        (0, 2): np.array([
            [2, -2],
            [-4, 5],
            [4, -8]
        ]),
        (0, 3): np.array([
            [1, -1],
            [8, 2],
            [6, 6]
        ]),
        (0, 4): np.array([
            [0, 2, 0],
            [2, 1, 1],
            [1, 3, -5]
        ]),
        (0, 5): np.array([
            [0, 2],
            [6, 0],
            [-2, -1]
        ]),
        (1, 0): np.array([
            [0, 6, -6],
            [0, -5, 0]
        ]),
        (1, 2): np.array([
            [0, -5],
            [-3, 5]
        ]),
        (1, 3): np.array([
            [2, 4],
            [2, 6]
        ]),
        (1, 4): np.array([
            [0, -6, 0],
            [0, -3, -9],
        ]),
        (1, 5): np.array([
            [1, 1],
            [-3, 0]
        ]),
        (2, 0): np.array([
            [0, 5, -5],
            [3, -6, 6],
        ]),
        (2, 1): np.array([
            [0, 5],
            [5, -4]
        ]),
        (2, 3): np.array([
            [0, 3],
            [-6, 5]
        ]),
        (2, 4): np.array([
            [2, -3, 0],
            [3, -4, -7]
        ]),
        (2, 5): np.array([
            [1, 1],
            [-3, 1]
        ]),
        (3, 0): np.array([
            [1, -6, -8],
            [0, 4, -8],
        ]),
        (3, 1): np.array([
            [3, 0],
            [-5, -6]
        ]),
        (3, 2): np.array([
            [0, 4],
            [-3, 3]
        ]),
        (3, 4): np.array([
            [4, 1, -3],
            [3, -2, -4]
        ]),
        (3, 5): np.array([
            [0, -3],
            [2, -3]
        ]),
        (4, 0): np.array([
            [2, 0, -3],
            [-1, 0, -3],
            [2, 2, 2]
        ]),
        (4, 1): np.array([
            [0, 0],
            [3, 1],
            [0, 9]
        ]),
        (4, 2): np.array([
            [-1, -4],
            [2, 5],
            [0, 3]
        ]),
        (4, 3): np.array([
            [-3, 0],
            [-3, 2],
            [-3, 2]
        ]),
        (4, 5): np.array([
            [-2, 0],
            [0, 0],
            [0, 0]
        ]),
        (5, 0): np.array([
            [5, -2, 5],
            [4, 4, 4]
        ]),
        (5, 1): np.array([
            [0, 5],
            [0, 0]
        ]),
        (5, 2): np.array([
            [0, 5],
            [-2, -2]
        ]),
        (5, 3): np.array([
            [0, 0],
            [2, 2]
        ]),
        (5, 4): np.array([
            [2, 0, 2],
            [-1, 0, 1]
        ])
    }
    polymg = PolymatrixGame(
        6,
        [3, 2, 2, 2, 3, 2],
        polymatrix
    )
    nfg = polymg.to_nf()
    ne = polym_lcp_solver(polymg)
    worked = nfg.is_nash(ne, tol=1e-5)
    assert_(worked)
