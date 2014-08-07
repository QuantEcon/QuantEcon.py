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
from .discrete_rv import DiscreteRV


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
        An array with invariant distributions as columns
    ergodic_sets : list(lists(int))
        A list of lists where each list in the main list
        has one of the ergodic sets.


    Methods
    -------
    invariant_distributions : This method finds invariant
                              distributions

    simulate_chain : Simulates the markov chain for a given
                     initial distribution
    """

    def __init__(self, P, pi0=None):
        self.P = P
        n, m = P.shape
        self.n = n

        if pi0 is None:
            self.pi0 = np.ones(n)/n

        else:
            self.pi0 = pi0

        # Check Properties
        # double check that P is a square matrix
        if n != m:
            raise ValueError('The transition matrix must be square!')

        # Double check that the rows of P sum to one
        if np.all(np.sum(P, axis=1) != np.ones(P.shape[0])):
            raise ValueError('The rows must sum to 1. P is a trans matrix')

    def __repr__(self):
        msg = "Markov process with transition matrix \n P = \n {0}"
        return msg.format(self.P)

    def __str__(self):
        return str(self.__repr__)

    def find_invariant_distributions(self, precision=None, tol=None):
        """
        This method computes the stationary distributions of P.
        These are the eigenvectors that correspond to the unit eigen-
        values of the matrix P' (They satisfy pi_{t+1}' = pi_{t}' P).  It
        simply calls the outer function mc_compute_stationary

        Parameters
        ----------
        precision : scalar(int), optional(default=None)
            Specifies the precision(number of digits of precision) with
            which to calculate the eigenvalues.  Unless your matrix has
            multiple eigenvalues that are near unity then no need to
            worry about this.
        tol : scalar(float), optional(default=None)
            Specifies the bandwith of eigenvalues to consider equivalent
            to unity.  It will consider all eigenvalues in [1-tol,
            1+tol] to be 1.  If tol is None then will use 2*1e-
            precision.  Only used if precision is defined

        Returns
        -------
        stat_dists : np.ndarray : float
            This is an array that has the stationary distributions as
            its columns.

        absorb_states : np.ndarray : ints
            This is a vector that says which of the states are
            absorbing states

        """
        P = self.P

        invar_dists = mc_compute_stationary(P, precision=precision, tol=tol)

        # Check to make sure all of the elements of invar_dist are positive
        if np.any(invar_dists<-1e-16):
            # print("Elements of your invariant distribution were negative; " +
            #       "trying with additional precision")

            if precision is None:
                invar_dists = mc_compute_stationary(P, precision=18, tol=tol)

            elif precision is not None:
                raise ValueError("Elements of your invariant distribution were" +
                                 "negative.  Try computing with higher precision")

        self.invar_dists = invar_dists

        return invar_dists.squeeze()

    def simulate_markov(self, init=0, sample_size=1000):
        """
        This method takes an initial distribution (or state) and
        simulates the markov chain with transition matrix P (defined by
        class) and initial distrubution init.  See mc_sample_path.

        Parameters
        ----------
        P : array_like(float, ndim=2)
            A discrete Markov transition matrix

        init : array_like(float ndim=1) or scalar(int)
            If init is an array_like then it is treated as the initial
            distribution across states.  If init is a scalar then it
            treated as the deterministic initial state.

        sample_size : scalar(int), optional(default=1000)
            The length of the sample path.

        Returns
        -------
        sim : array_like(int, ndim=1)
            The simulation of states
        """
        sim = mc_sample_path(self.P, init, sample_size)

        return sim


def mc_compute_stationary(P, precision=None, tol=None):
    """
    Computes the stationary distribution of Markov matrix P.

    Parameters
    ----------
    P : array_like(float, ndim=2)
        A discrete Markov transition matrix
    precision : scalar(int), optional(default=None)
        Specifies the precision(number of digits of precision) with
        which to calculate the eigenvalues.  Unless your matrix has
        multiple eigenvalues that are near unity then no need to worry
        about this.
    tol : scalar(float), optional(default=None)
        Specifies the bandwith of eigenvalues to consider equivalent to
        unity.  It will consider all eigenvalues in [1-tol, 1+tol] to be
        1.  If tol is None then will use 2*1e-precision.  Only used if
        precision is defined

    Returns
    -------
    solution : array_like(float, ndim=2)
        The stationary distributions for P

    """
    n = P.shape[0]

    if precision is None:
        # Compute eigenvalues and eigenvectors
        eigvals, eigvecs = la.eig(P, left=True, right=False)

        # Find the index for unit eigenvalues
        index = np.where(abs(eigvals - 1.) < 1e-12)[0]

        # Pull out the eigenvectors that correspond to unit eig-vals
        uniteigvecs = eigvecs[:, index]

        invar_dists = uniteigvecs/np.sum(uniteigvecs, axis=0)

        # Since we will be accessing the columns of this matrix, we
        # might consider adding .astype(np.float, order='F') to make it
        # column major at beginning
        return invar_dists

    else:
        # Create a list to store eigvals
        invar_dists_list = []
        if tol is None:
            # If tolerance isn't specified then use 2*precision
            tol = 2 * 10**(-precision + 1)

        with mp.workdps(precision):
            eigvals, eigvecs = mp.eig(mp.matrix(P), left=True, right=False)

            for ind, el in enumerate(eigvals):
                if el>=(mp.mpf(1)-mp.mpf(tol)) and el<=(mp.mpf(1)+mp.mpf(tol)):
                    invar_dists_list.append(eigvecs[ind, :])

            invar_dists = np.asarray(invar_dists_list).T
            invar_dists = (invar_dists/sum(invar_dists)).astype(np.float)

        return invar_dists.squeeze()


def mc_sample_path(P, init=0, sample_size=1000):
    """
    Generates one sample path from a finite Markov chain with (n x n)
    Markov matrix P on state space S = {0,...,n-1}.

    Parameters
    ----------
    P : array_like(float, ndim=2)
        A discrete Markov transition matrix

    init : array_like(float ndim=1) or scalar(int)
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