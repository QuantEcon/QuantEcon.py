"""
Filename: random.py

Author: Daisuke Oyama

Generate a MarkovChain randomly.

"""
import numpy as np
import scipy.sparse

from ..util import check_random_state, numba_installed, jit
from ..random import (
    probvec, sample_without_replacement
)

from .core import MarkovChain

def random_markov_chain(n, k=None, sparse=False, random_state=None):
    """
    Return a randomly sampled MarkovChain instance with n states, where
    each state has k states with positive transition probability.

    Parameters
    ----------
    n : scalar(int)
        Number of states.

    k : scalar(int), optional(default=None)
        Number of states that may be reached from each state with
        positive probability. Set to n if not specified.

    sparse : bool, optional(default=False)
        Whether to store the transition probability matrix in sparse
        matrix form. (Sparse format is not yet implemented.)

    random_state : scalar(int) or np.random.RandomState,
                   optional(default=None)
        Random seed (integer) or np.random.RandomState instance to set
        the initial state of the random number generator for
        reproducibility. If None, a randomly initialized RandomState is
        used.

    Returns
    -------
    mc : MarkovChain

    Examples
    --------
    >>> mc = qe.markov.random_markov_chain(3, random_state=1234)
    >>> mc.P
    array([[ 0.19151945,  0.43058932,  0.37789123],
           [ 0.43772774,  0.34763084,  0.21464142],
           [ 0.27259261,  0.5073832 ,  0.22002419]])
    >>> mc = qe.markov.random_markov_chain(3, k=2, random_state=1234)
    >>> mc.P
    array([[ 0.19151945,  0.80848055,  0.        ],
           [ 0.        ,  0.62210877,  0.37789123],
           [ 0.56227226,  0.        ,  0.43772774]])

    """
    if sparse:
        raise NotImplementedError
    P = random_stochastic_matrix(n, k, sparse, format='csr',
                                 random_state=random_state)
    mc = MarkovChain(P)
    return mc


def random_stochastic_matrix(n, k=None, sparse=False, format='csr',
                             random_state=None):
    """
    Return a randomly sampled n x n stochastic matrix with k nonzero
    entries for each row.

    Parameters
    ----------
    n : scalar(int)
        Number of states.

    k : scalar(int), optional(default=None)
        Number of nonzero entries in each row of the matrix. Set to n if
        not specified.

    sparse : bool, optional(default=False)
        Whether to generate the matrix in sparse matrix form.

    format : str in {'bsr', 'csr', 'csc', 'coo', 'lil', 'dia', 'dok'},
             optional(default='csr')
        Sparse matrix format. Relevant only when sparse=True.

    random_state : scalar(int) or np.random.RandomState,
                   optional(default=None)
        Random seed (integer) or np.random.RandomState instance to set
        the initial state of the random number generator for
        reproducibility. If None, a randomly initialized RandomState is
        used.

    Returns
    -------
    P : numpy ndarray or scipy sparse matrix (float, ndim=2)
        Stochastic matrix.

    See also
    --------
    random_markov_chain : Return a random MarkovChain instance.

    """
    if not (isinstance(n, int) and n > 0):
        raise ValueError('n must be a positive integer')
    if k is None:
        k = n
    if not (isinstance(k, int) and 0 < k <= n):
        raise ValueError('k must be an integer with 0 < k <= n')

    # n prob vectors of dimension k, shape (n, k)
    probvecs = probvec(n, k, random_state=random_state)

    if k == n:
        P = probvecs
        if sparse:
            return scipy.sparse.coo_matrix(P).asformat(format)
        else:
            return P

    # if k < n:
    rows = np.repeat(np.arange(n), k)
    cols = \
        sample_without_replacement(
            n, k, num_trials=n, random_state=random_state
        ).ravel()
    data = probvecs.ravel()

    if sparse:
        P = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(n, n))
        return P.asformat(format)
    else:
        P = np.zeros((n, n))
        P[rows, cols] = data
        return P
