# flake8: noqa
"""
Import the main names to top level.
"""

__version__ = '0.7.2'

try:
    import numba
except:
    raise ImportError(
        "Cannot import numba from current anaconda distribution. \
            Please run `conda install numba` to install the latest version.")

#-Modules-#
from . import distributions
from . import game_theory
from . import quad
from . import random
from . import optimize

#-Objects-#
from ._compute_fp import compute_fixed_point
from ._discrete_rv import DiscreteRV
from ._dle import DLE
from ._ecdf import ECDF
from ._estspec import smooth, periodogram, ar_periodogram
# from .game_theory import <objects-here>                           #Place Holder if we wish to promote any general objects to the qe namespace.
from ._graph_tools import DiGraph, random_tournament_graph
from ._gridtools import (
    cartesian, mlinspace, cartesian_nearest_index, simplex_grid, simplex_index,
    num_compositions
)
from ._inequality import lorenz_curve, gini_coefficient, shorrocks_index, \
    rank_size
from ._kalman import Kalman
from ._lae import LAE
from ._arma import ARMA
from ._lqcontrol import LQ, LQMarkov
from ._filter import hamilton_filter
from ._lqnash import nnash
from ._ivp import IVP
from ._lss import LinearStateSpace
from ._matrix_eqn import solve_discrete_lyapunov, solve_discrete_riccati
from ._quadsums import var_quadratic_sum, m_quadratic_sum
#->Propose Delete From Top Level
#Promote to keep current examples working
from .markov import MarkovChain, random_markov_chain, random_stochastic_matrix, \
    gth_solve, tauchen, rouwenhorst, estimate_mc
#Imports that Should be Deprecated with markov package
from .markov import mc_compute_stationary, mc_sample_path
#<-
from ._rank_nullspace import rank_est, nullspace
from ._robustlq import RBLQ
from .util import searchsorted, fetch_nb_dependencies, tic, tac, toc
