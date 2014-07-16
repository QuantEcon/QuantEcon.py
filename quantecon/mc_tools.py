"""
Filename: mc_tools.py

Authors: Thomas J. Sargent, John Stachurski

"""
import numpy as np
from discrete_rv import DiscreteRV


def mc_compute_stationary(P):
    """
    Computes the stationary distribution of Markov matrix P.

    Parameters
    ----------
    P : array_like(float, ndim=2)
        A discrete Markov transition matrix

    Returns
    -------
    solution : array_like(float, ndim=1)
        The stationary distribution for P

    Note: Currently only supports transition matrices with a unique
    invariant distribution.  See issue 19.

    """
    n = len(P)                               # P is n x n
    I = np.identity(n)                       # Identity matrix
    B, b = np.ones((n, n)), np.ones((n, 1))  # Matrix and vector of ones
    A = np.transpose(I - P + B)
    solution = np.linalg.solve(A, b).flatten()

    return solution


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

