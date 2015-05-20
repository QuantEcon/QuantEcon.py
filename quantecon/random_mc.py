"""
Filename: random_mc.py

Author: Daisuke Oyama

Generate a MarkovChain randomly.

"""
import numpy as np
import scipy.sparse

from .mc_tools import MarkovChain
from .external import numba_installed, jit


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

    random_state : {numpy.random.RandomState, int},
                   optional(default=None)
        Random number generator instance or random seed (int).

    Returns
    -------
    mc : MarkovChain

    Examples
    --------
    >>> mc = qe.random_markov_chain(3, random_state=1234)
    >>> mc.P
    array([[ 0.19151945,  0.43058932,  0.37789123],
           [ 0.43772774,  0.34763084,  0.21464142],
           [ 0.27259261,  0.5073832 ,  0.22002419]])
    >>> mc = qe.random_markov_chain(3, k=2, random_state=1234)
    >>> mc.P
    array([[ 0.        ,  0.80848055,  0.19151945],
           [ 0.62210877,  0.        ,  0.37789123],
           [ 0.        ,  0.56227226,  0.43772774]])

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

    random_state : {numpy.random.RandomState, int},
                   optional(default=None)
        Random number generator instance or random seed (int).

    Returns
    -------
    P : numpy ndarray or scipy sparse matrix (float, ndim=2)
        Stochastic matrix.

    """
    if not (isinstance(n, int) and n > 0):
        raise ValueError('n must be a positive integer')
    if k is None:
        k = n
    if not (isinstance(k, int) and 0 < k <= n):
        raise ValueError('k must be an integer with 0 < k <= n')

    if random_state is None or isinstance(random_state, int):
        _random_state = np.random.RandomState(random_state)
    elif isinstance(random_state, np.random.RandomState):
        _random_state = random_state
    else:
        raise ValueError

    # n prob vectors of dimension k, shape (n, k)
    probvecs = random_probvec(n, k, random_state=_random_state)

    if k == n:
        P = probvecs
        if sparse:
            return scipy.sparse.coo_matrix(P).asformat(format)
        else:
            return P

    # if k < n:
    rows = np.repeat(np.arange(n), k)
    cols = \
        random_choice_without_replacement(
            n, k, num_trials=n, random_state=_random_state
        ).ravel()
    data = probvecs.ravel()

    if sparse:
        P = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(n, n))
        return P.asformat(format)
    else:
        P = np.zeros((n, n))
        P[rows, cols] = data
        return P


def random_probvec(m, k, random_state=None):
    """
    Return m randomly sampled probability vectors of dimension k.

    Parameters
    ----------
    m : scalar(int)
        Number of probability vectors.

    k : scalar(int)
        Dimension of each probability vectors.

    random_state : numpy.random.RandomState, optional(default=None)
        Random number generator. If None, np.random is used.

    Returns
    -------
    ndarray(float, ndim=2)
        Array of shape (m, k) containing probability vectors as rows.

    """
    x = np.empty((m, k+1))

    if random_state is None:
        random_state = np.random
    r = random_state.random_sample(size=(m, k-1))

    r.sort(axis=-1)
    x[:, 0], x[:, 1:k], x[:, k] = 0, r, 1
    return np.diff(x, axis=-1)


def random_choice_without_replacement(n, k, num_trials=None,
                                      random_state=None):
    """
    Randomly choose k integers without replacement from 0, ..., n-1.

    Parameters
    ----------
    n : scalar(int)
        Number of integers, 0, ..., n-1, to sample from.

    k : scalar(int)
        Number of integers to sample.

    num_trials : scalar(int), optional(default=None)
        Number of trials.

    random_state : numpy.random.RandomState, optional(default=None)
        Random number generator. If None, np.random is used.

    Returns
    -------
    result : ndarray(int, ndim=1 or 2)
        Array of shape (k,) if num_trials is None, or of shape
        (num_trials, k) otherwise, (each row of) which contains k unique
        random elements chosen from 0, ..., n-1.

    Examples
    --------
    >>> qe.random_mc.random_choice_without_replacement(5, 3)
    array([1, 4, 3])
    >>> qe.random_mc.random_choice_without_replacement(5, 3, num_trials=4)
    array([[1, 2, 0],
           [2, 0, 4],
           [4, 1, 2],
           [2, 3, 4]])

    """
    if n <= 0:
        raise ValueError('n must be greater than 0')
    if k > n:
        raise ValueError('k must be smaller than or equal to n')

    m = 1 if num_trials is None else num_trials

    if random_state is None:
        random_state = np.random
    r = random_state.random_sample(size=(m, k))

    # Logic taken from random.sample in the standard library
    result = np.empty((m, k), dtype=int)
    pool = np.empty((m, n), dtype=int)
    for i in range(m):
        for j in range(n):
            pool[i, j] = j

    for i in range(m):
        for j in range(k):
            idx = int(np.floor(r[i, j] * (n-j)))  # np.floor returns a float
            result[i, j] = pool[i, idx]
            pool[i, idx] = pool[i, n-j-1]

    if num_trials is None:
        return result[0]
    else:
        return result

if numba_installed:
    random_choice_without_replacement = jit(random_choice_without_replacement)
