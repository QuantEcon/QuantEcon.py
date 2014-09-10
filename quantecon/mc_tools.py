"""
Authors: Chase Coleman, Spencer Lyon, Daisuke Oyama, Tom Sargent,
         John Stachurski

Filename: mc_tools.py

This file contains some useful objects for handling a discrete markov
transition matrix.  It contains code written by several people and
was ultimately compiled into a single file to take advantage of the
pros of each.

"""
from __future__ import division
import numpy as np
import scipy.linalg as la
import sympy.mpmath as mp
import sys
from .discrete_rv import DiscreteRV
from .stochmatrix import StochMatrix, stationary_dists
from warnings import warn


class DMarkov(object):
    """
    This class is used as a container for a discrete Markov transition
    matrix or a discrete Markov chain.  It stores useful information
    such as the stationary distributions and allows simulation using a
    specified initial distribution.


    Parameters
    ----------
    P : array_like(float, ndim=2)
        The transition matrix.  Must be of shape n x n.

    pi_0 : array_like(float, ndim=1), optional(default=None)
        The initial probability distribution.  If no intial distribution
        is specified, then it will be a distribution with equally
        probability on each state


    Attributes
    ----------
    P : array_like(float, ndim=2)
        The transition matrix.  Must be of shape n x n.

    pi_0 : array_like(float, ndim=1)
        The initial probability distribution.

    stationary_dists : array_like(float, ndim=2)
        An array with stationary distributions as rows.

    is_irreducible : bool
        Indicate whether the array is an irreducible matrix.

    num_comm_classes : int
        Number of communication classes.

    comm_classes : list(list(int))
        List of lists containing the communication classes.

    num_rec_classes : int
        Number of recurrent classes.

    rec_classes : list(list(int))
        List of lists containing the recurrent classes.

    Methods
    -------
    compute_stationary : Finds stationary distributions

    simulate : Simulates the markov chain for a given initial
        distribution
    """

    def __init__(self, P, pi_0=None):
        self.P = StochMatrix(P)
        n, m = self.P.shape
        self.n = n

        if pi_0 is None:
            self.pi_0 = np.ones(n)/n

        else:
            self.pi_0 = pi_0

        self.stationary_dists = None

        # Check Properties
        # double check that P is a square matrix
        if n != m:
            raise ValueError('The transition matrix must be square!')

        # Double check that the rows of P sum to one
        if np.any(np.sum(self.P, axis=1) != np.ones(self.P.shape[0])):
            raise ValueError('The rows must sum to 1. P is a trans matrix')

    def __repr__(self):
        msg = "Markov process with transition matrix \nP = \n{0}"

        if self.stationary_dists is None:
            return msg.format(self.P)
        else:
            msg = msg + "\nand stationary distributions \n{1}"
            return msg.format(self.P, self.stationary_dists)

    def __str__(self):
        return str(self.__repr__)

    def compute_stationary(self):
        """
        Computes the stationary distributions of P. These are the left
        eigenvectors that correspond to the unit eigen-values of the
        matrix P' (They satisfy pi_{{t+1}}' = pi_{{t}}' P). It simply
        calls the outer function stationary_dists.

        Returns
        -------
        stationary_dists : array_like(float, ndim=2)
            This is an array that has the stationary distributions as
            its rows.

        """
        self.stationary_dists = stationary_dists(self.P)

        return self.stationary_dists

    def simulate(self, init=0, sample_size=1000):
        sim = mc_sample_path(self.P, init, sample_size)

        return sim

    @property
    def is_irreducible(self):
        return self.P.is_irreducible

    @property
    def num_comm_classes(self):
        return self.P.num_comm_classes

    @property
    def comm_classes(self):
        return self.P.comm_classes()

    @property
    def num_rec_classes(self):
        return self.P.num_rec_classes

    @property
    def rec_classes(self):
        return self.P.rec_classes()


def mc_sample_path(P, init=0, sample_size=1000):
    # === set up array to store output === #
    X = np.empty(sample_size, dtype=int)
    if isinstance(init, int):
        X[0] = init
    else:
        X[0] = DiscreteRV(init).draw()

    # === turn each row into a distribution === #
    # In particular, let P_dist[i] be the distribution corresponding to the
    # i-th row P[i,:]
    n = len(P)
    P_dist = [DiscreteRV(P[i,:]) for i in range(n)]

    # === generate the sample path === #
    for t in range(sample_size - 1):
        X[t+1] = P_dist[X[t]].draw()

    return X


#---------------------------------------------------------------------#
# Set up the docstrings for the functions
#---------------------------------------------------------------------#

# For drawing a sample path
_sample_path_docstr = \
"""
Generates one sample path from a finite Markov chain with (n x n)
Markov matrix P on state space S = {{0,...,n-1}}.

Parameters
----------
{p_arg}init : array_like(float ndim=1) or scalar(int)
    If init is an array_like then it is treated as the initial
    distribution across states.  If init is a scalar then it
    treated as the deterministic initial state.

sample_size : scalar(int), optional(default=1000)
    The length of the sample path.

Returns
-------
X : array_like(int, ndim=1)
    The simulation of states

"""

# set docstring for functions
mc_sample_path.__doc__ = _sample_path_docstr.format(p_arg=
"""P : array_like(float, ndim=2)
    A discrete Markov transition matrix

""")

# set docstring for methods

if sys.version_info[0] == 3:
    DMarkov.simulate.__doc__ = _sample_path_docstr.format(p_arg="")
elif sys.version_info[0] == 2:
    DMarkov.simulate.__func__.__doc__ = _sample_path_docstr.format(p_arg="")
