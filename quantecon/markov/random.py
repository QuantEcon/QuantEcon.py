"""
Generate MarkovChain and DiscreteDP instances randomly.

"""
import numpy as np
import scipy.sparse

from .core import MarkovChain
from .ddp import DiscreteDP
from .utilities import sa_indices
from ..util import check_random_state
from ..random import probvec, sample_without_replacement


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
        matrix form.

    random_state : int or np.random.RandomState/Generator, optional
        Random seed (integer) or np.random.RandomState or Generator
        instance to set the initial state of the random number generator
        for reproducibility. If None, a randomly initialized RandomState
        is used.

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

    format : str, optional(default='csr')
        Sparse matrix format, str in {'bsr', 'csr', 'csc', 'coo', 'lil',
        'dia', 'dok'}. Relevant only when sparse=True.

    random_state : int or np.random.RandomState/Generator, optional
        Random seed (integer) or np.random.RandomState or Generator
        instance to set the initial state of the random number generator
        for reproducibility. If None, a randomly initialized RandomState
        is used.

    Returns
    -------
    P : numpy ndarray or scipy sparse matrix (float, ndim=2)
        Stochastic matrix.

    See also
    --------
    random_markov_chain : Return a random MarkovChain instance.

    """
    P = _random_stochastic_matrix(m=n, n=n, k=k, sparse=sparse, format=format,
                                  random_state=random_state)
    return P


def _random_stochastic_matrix(m, n, k=None, sparse=False, format='csr',
                              random_state=None):
    """
    Generate a "non-square stochastic matrix" of shape (m, n), which
    contains as rows m probability vectors of length n with k nonzero
    entries.

    For other parameters, see `random_stochastic_matrix`.

    """
    if k is None:
        k = n
    # m prob vectors of dimension k, shape (m, k)
    probvecs = probvec(m, k, random_state=random_state)

    if k == n:
        P = probvecs
        if sparse:
            return scipy.sparse.coo_matrix(P).asformat(format)
        else:
            return P

    # if k < n:
    rows = np.repeat(np.arange(m), k)
    cols = \
        sample_without_replacement(
            n, k, num_trials=m, random_state=random_state
        ).ravel()
    data = probvecs.ravel()

    if sparse:
        P = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(m, n))
        return P.asformat(format)
    else:
        P = np.zeros((m, n))
        P[rows, cols] = data
        return P


def random_discrete_dp(num_states, num_actions, beta=None,
               k=None, scale=1, sparse=False, sa_pair=False,
               random_state=None):
    """
    Generate a DiscreteDP randomly. The reward values are drawn from the
    normal distribution with mean 0 and standard deviation `scale`.

    Parameters
    ----------
    num_states : scalar(int)
        Number of states.

    num_actions : scalar(int)
        Number of actions.

    beta : scalar(float), optional(default=None)
        Discount factor. Randomly chosen from [0, 1) if not specified.

    k : scalar(int), optional(default=None)
        Number of possible next states for each state-action pair. Equal
        to `num_states` if not specified.

    scale : scalar(float), optional(default=1)
        Standard deviation of the normal distribution for the reward
        values.

    sparse : bool, optional(default=False)
        Whether to store the transition probability array in sparse
        matrix form.

    sa_pair : bool, optional(default=False)
        Whether to represent the data in the state-action pairs
        formulation. (If `sparse=True`, automatically set `True`.)

    random_state : int or np.random.RandomState/Generator, optional
        Random seed (integer) or np.random.RandomState or Generator
        instance to set the initial state of the random number generator
        for reproducibility. If None, a randomly initialized RandomState
        is used.

    Returns
    -------
    ddp : DiscreteDP
        An instance of DiscreteDP.

    """
    if sparse:
        sa_pair = True

    # Number of state-action pairs
    L = num_states * num_actions

    random_state = check_random_state(random_state)
    R = scale * random_state.standard_normal(L)
    Q = _random_stochastic_matrix(L, num_states, k=k,
                                  sparse=sparse, format='csr',
                                  random_state=random_state)
    if beta is None:
        beta = random_state.random()

    if sa_pair:
        s_indices, a_indices = sa_indices(num_states, num_actions)
    else:
        s_indices, a_indices = None, None
        R.shape = (num_states, num_actions)
        Q.shape = (num_states, num_actions, num_states)

    ddp = DiscreteDP(R, Q, beta, s_indices, a_indices)
    return ddp
