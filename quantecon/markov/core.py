r"""
This file contains some useful objects for handling a finite-state
discrete-time Markov chain.

Definitions and Some Basic Facts about Markov Chains
----------------------------------------------------

Let :math:`\{X_t\}` be a Markov chain represented by an :math:`n \times
n` stochastic matrix :math:`P`. State :math:`i` *has access* to state
:math:`j`, denoted :math:`i \to j`, if :math:`i = j` or :math:`P^k[i, j]
> 0` for some :math:`k = 1, 2, \ldots`; :math:`i` and `j` *communicate*,
denoted :math:`i \leftrightarrow j`, if :math:`i \to j` and :math:`j \to
i`. The binary relation :math:`\leftrightarrow` is an equivalent
relation. A *communication class* of the Markov chain :math:`\{X_t\}`,
or of the stochastic matrix :math:`P`, is an equivalent class of
:math:`\leftrightarrow`. Equivalently, a communication class is a
*strongly connected component* (SCC) in the associated *directed graph*
:math:`\Gamma(P)`, a directed graph with :math:`n` nodes where there is
an edge from :math:`i` to :math:`j` if and only if :math:`P[i, j] > 0`.
The Markov chain, or the stochastic matrix, is *irreducible* if it
admits only one communication class, or equivalently, if
:math:`\Gamma(P)` is *strongly connected*.

A state :math:`i` is *recurrent* if :math:`i \to j` implies :math:`j \to
i`; it is *transient* if it is not recurrent. For any :math:`i, j`
contained in a communication class, :math:`i` is recurrent if and only
if :math:`j` is recurrent. Therefore, recurrence is a property of a
communication class. Thus, a communication class is a *recurrent class*
if it contains a recurrent state. Equivalently, a recurrent class is a
SCC that corresponds to a sink node in the *condensation* of the
directed graph :math:`\Gamma(P)`, where the condensation of
:math:`\Gamma(P)` is a directed graph in which each SCC is replaced with
a single node and there is an edge from one SCC :math:`C` to another SCC
:math:`C'` if :math:`C \neq C'` and there is an edge from some node in
:math:`C` to some node in :math:`C'`. A recurrent class is also called a
*closed communication class*. The condensation is acyclic, so that there
exists at least one recurrent class.

For example, if the entries of :math:`P` are all strictly positive, then
the whole state space is a communication class as well as a recurrent
class. (More generally, if there is only one communication class, then
it is a recurrent class.) As another example, consider the stochastic
matrix :math:`P = [[1, 0], [0,5, 0.5]]`. This has two communication
classes, :math:`\{0\}` and :math:`\{1\}`, and :math:`\{0\}` is the only
recurrent class.

A *stationary distribution* of the Markov chain :math:`\{X_t\}`, or of
the stochastic matrix :math:`P`, is a nonnegative vector :math:`x` such
that :math:`x' P = x'` and :math:`x' \mathbf{1} = 1`, where
:math:`\mathbf{1}` is the vector of ones. The Markov chain has a unique
stationary distribution if and only if it has a unique recurrent class.
More generally, each recurrent class has a unique stationary
distribution whose support equals that recurrent class. The set of all
stationary distributions is given by the convex hull of these unique
stationary distributions for the recurrent classes.

A natural number :math:`d` is the *period* of state :math:`i` if it is
the greatest common divisor of all :math:`k`'s such that :math:`P^k[i,
i] > 0`; equivalently, it is the GCD of the lengths of the cycles in
:math:`\Gamma(P)` passing through :math:`i`. For any :math:`i, j`
contained in a communication class, :math:`i` has period :math:`d` if
and only if :math:`j` has period :math:`d`. The *period* of an
irreducible Markov chain (or of an irreducible stochastic matrix) is the
period of any state. We define the period of a general (not necessarily
irreducible) Markov chain to be the least common multiple of the periods
of its recurrent classes, where the period of a recurrent class is the
period of any state in that class. A Markov chain is *aperiodic* if its
period is one. A Markov chain is irreducible and aperiodic if and only
if it is *uniformly ergodic*, i.e., there exists some :math:`m` such
that :math:`P^m[i, j] > 0` for all :math:`i, j` (in this case, :math:`P`
is also called *primitive*).

Suppose that an irreducible Markov chain has period :math:`d`. Fix any
state, say state :math:`0`. For each :math:`m = 0, \ldots, d-1`, let
:math:`S_m` be the set of states :math:`i` such that :math:`P^{kd+m}[0,
i] > 0` for some :math:`k`. These sets :math:`S_0, \ldots, S_{d-1}`
constitute a partition of the state space and are called the *cyclic
classes*. For each :math:`S_m` and each :math:`i \in S_m`, we have
:math:`\sum_{j \in S_{m+1}} P[i, j] = 1`, where :math:`S_d = S_0`.

"""
import numbers
from math import gcd
import numpy as np
from scipy import sparse
from numba import jit

from .gth_solve import gth_solve
from .._graph_tools import DiGraph
from ..util import searchsorted, check_random_state, rng_integers


class MarkovChain:
    """
    Class for a finite-state discrete-time Markov chain. It stores
    useful information such as the stationary distributions, and
    communication, recurrent, and cyclic classes, and allows simulation
    of state transitions.

    Parameters
    ----------
    P : array_like or scipy sparse matrix (float, ndim=2)
        The transition matrix.  Must be of shape n x n.

    state_values : array_like(default=None)
        Array_like of length n containing the values associated with the
        states, which must be homogeneous in type. If None, the values
        default to integers 0 through n-1.

    Attributes
    ----------
    P : ndarray or scipy.sparse.csr_matrix (float, ndim=2)
        See Parameters

    stationary_distributions : array_like(float, ndim=2)
        Array containing stationary distributions, one for each
        recurrent class, as rows.

    is_irreducible : bool
        Indicate whether the Markov chain is irreducible.

    num_communication_classes : int
        The number of the communication classes.

    communication_classes_indices : list(ndarray(int))
        List of numpy arrays containing the indices of the communication
        classes.

    communication_classes : list(ndarray)
        List of numpy arrays containing the communication classes, where
        the states are annotated with their values (if `state_values` is
        not None).

    num_recurrent_classes : int
        The number of the recurrent classes.

    recurrent_classes_indices : list(ndarray(int))
        List of numpy arrays containing the indices of the recurrent
        classes.

    recurrent_classes : list(ndarray)
        List of numpy arrays containing the recurrent classes, where the
        states are annotated with their values (if `state_values` is not
        None).

    is_aperiodic : bool
        Indicate whether the Markov chain is aperiodic.

    period : int
        The period of the Markov chain.

    cyclic_classes_indices : list(ndarray(int))
        List of numpy arrays containing the indices of the cyclic
        classes. Defined only when the Markov chain is irreducible.

    cyclic_classes : list(ndarray)
        List of numpy arrays containing the cyclic classes, where the
        states are annotated with their values (if `state_values` is not
        None). Defined only when the Markov chain is irreducible.

    Notes
    -----
    In computing stationary distributions, if the input matrix is a
    sparse matrix, internally it is converted to a dense matrix.

    """

    def __init__(self, P, state_values=None):
        if sparse.issparse(P):  # Sparse matrix
            self.P = sparse.csr_matrix(P)
            self.is_sparse = True
        else:  # Dense matrix
            self.P = np.asarray(P)
            self.is_sparse = False

        # Check Properties
        # Double check that P is a square matrix
        if len(self.P.shape) != 2 or self.P.shape[0] != self.P.shape[1]:
            raise ValueError('P must be a square matrix')

        # The number of states
        self.n = self.P.shape[0]

        # Double check that P is a nonnegative matrix
        if not self.is_sparse:
            data_nonnegative = (self.P >= 0)  # ndarray
        else:
            data_nonnegative = (self.P.data >= 0)  # csr_matrx
        if not np.all(data_nonnegative):
            raise ValueError('P must be nonnegative')

        # Double check that the rows of P sum to one
        row_sums = self.P.sum(axis=1)
        if self.is_sparse:  # row_sums is np.matrix (ndim=2)
            row_sums = row_sums.getA1()
        if not np.allclose(row_sums, np.ones(self.n)):
            raise ValueError('The rows of P must sum to 1')

        # Call the setter method
        self.state_values = state_values

        # To analyze the structure of P as a directed graph
        self._digraph = None

        self._stationary_dists = None
        self._cdfs = None  # For dense matrix
        self._cdfs1d = None  # For sparse matrix

    def __repr__(self):
        msg = "Markov chain with transition matrix \nP = \n{0}"

        if self._stationary_dists is None:
            return msg.format(self.P)
        else:
            msg = msg + "\nand stationary distributions \n{1}"
            return msg.format(self.P, self._stationary_dists)

    def __str__(self):
        return str(self.__repr__)

    @property
    def state_values(self):
        return self._state_values

    @state_values.setter
    def state_values(self, values):
        if values is None:
            self._state_values = None
        else:
            values = np.asarray(values)
            if (values.ndim < 1) or (values.shape[0] != self.n):
                raise ValueError(
                    'state_values must be an array_like of length n'
                )
            if np.issubdtype(values.dtype, np.object_):
                raise ValueError(
                    'data in state_values must be homogeneous in type'
                )
            self._state_values = values

    def get_index(self, value):
        """
        Return the index (or indices) of the given value (or values) in
        `state_values`.

        Parameters
        ----------
        value
            Value(s) to get the index (indices) for.

        Returns
        -------
        idx : int or ndarray(int)
            Index of `value` if `value` is a single state value; array
            of indices if `value` is an array_like of state values.

        """
        if self.state_values is None:
            state_values_ndim = 1
        else:
            state_values_ndim = self.state_values.ndim

        values = np.asarray(value)

        if values.ndim <= state_values_ndim - 1:
            return self._get_index(value)
        elif values.ndim == state_values_ndim:  # array of values
            k = values.shape[0]
            idx = np.empty(k, dtype=int)
            for i in range(k):
                idx[i] = self._get_index(values[i])
            return idx
        else:
            raise ValueError('invalid value')

    def _get_index(self, value):
        """
        Return the index of the given value in `state_values`.

        Parameters
        ----------
        value
            Value to get the index for.

        Returns
        -------
        idx : int
            Index of `value`.

        """
        error_msg = 'value {0} not found'.format(value)

        if self.state_values is None:
            if isinstance(value, numbers.Integral) and (0 <= value < self.n):
                return value
            else:
                raise ValueError(error_msg)

        # if self.state_values is not None:
        if self.state_values.ndim == 1:
            try:
                idx = np.where(self.state_values == value)[0][0]
                return idx
            except IndexError:
                raise ValueError(error_msg)
        else:
            idx = 0
            while idx < self.n:
                if np.array_equal(self.state_values[idx], value):
                    return idx
                idx += 1
            raise ValueError(error_msg)

    @property
    def digraph(self):
        if self._digraph is None:
            self._digraph = DiGraph(self.P, node_labels=self.state_values)
        return self._digraph

    @property
    def is_irreducible(self):
        return self.digraph.is_strongly_connected

    @property
    def num_communication_classes(self):
        return self.digraph.num_strongly_connected_components

    @property
    def communication_classes_indices(self):
        return self.digraph.strongly_connected_components_indices

    @property
    def communication_classes(self):
        return self.digraph.strongly_connected_components

    @property
    def num_recurrent_classes(self):
        return self.digraph.num_sink_strongly_connected_components

    @property
    def recurrent_classes_indices(self):
        return self.digraph.sink_strongly_connected_components_indices

    @property
    def recurrent_classes(self):
        return self.digraph.sink_strongly_connected_components

    @property
    def is_aperiodic(self):
        if self.is_irreducible:
            return self.digraph.is_aperiodic
        else:
            return self.period == 1

    @property
    def period(self):
        if self.is_irreducible:
            return self.digraph.period
        else:
            # Determine the period, the LCM of the periods of rec_classes
            d = 1
            for rec_class in self.recurrent_classes_indices:
                period = self.digraph.subgraph(rec_class).period
                d = (d * period) // gcd(d, period)

            return d

    @property
    def cyclic_classes(self):
        if not self.is_irreducible:
            raise NotImplementedError(
                'Not defined for a reducible Markov chain'
            )
        else:
            return self.digraph.cyclic_components

    @property
    def cyclic_classes_indices(self):
        if not self.is_irreducible:
            raise NotImplementedError(
                'Not defined for a reducible Markov chain'
            )
        else:
            return self.digraph.cyclic_components_indices

    def _compute_stationary(self):
        """
        Store the stationary distributions in self._stationary_distributions.

        """
        if self.is_irreducible:
            if not self.is_sparse:  # Dense
                stationary_dists = gth_solve(self.P).reshape(1, self.n)
            else:  # Sparse
                stationary_dists = \
                    gth_solve(self.P.toarray(),
                              overwrite=True).reshape(1, self.n)
        else:
            rec_classes = self.recurrent_classes_indices
            stationary_dists = np.zeros((len(rec_classes), self.n))
            for i, rec_class in enumerate(rec_classes):
                P_rec_class = self.P[np.ix_(rec_class, rec_class)]
                if self.is_sparse:
                    P_rec_class = P_rec_class.toarray()
                stationary_dists[i, rec_class] = \
                    gth_solve(P_rec_class, overwrite=True)

        self._stationary_dists = stationary_dists

    @property
    def stationary_distributions(self):
        if self._stationary_dists is None:
            self._compute_stationary()
        return self._stationary_dists

    @property
    def cdfs(self):
        if (self._cdfs is None) and not self.is_sparse:
            # See issue #137#issuecomment-96128186
            cdfs = np.empty((self.n, self.n), order='C', dtype=self.P.dtype)
            np.cumsum(self.P, axis=-1, out=cdfs)
            self._cdfs = cdfs
        return self._cdfs

    @property
    def cdfs1d(self):
        if (self._cdfs1d is None) and self.is_sparse:
            data = self.P.data
            indptr = self.P.indptr

            cdfs1d = np.empty(self.P.nnz, order='C', dtype=data.dtype)
            for i in range(self.n):
                cdfs1d[indptr[i]:indptr[i+1]] = \
                    data[indptr[i]:indptr[i+1]].cumsum()
            self._cdfs1d = cdfs1d
        return self._cdfs1d

    def simulate_indices(self, ts_length, init=None, num_reps=None,
                         random_state=None):
        """
        Simulate time series of state transitions, where state indices
        are returned.

        Parameters
        ----------
        ts_length : scalar(int)
            Length of each simulation.

        init : int or array_like(int, ndim=1), optional
            Initial state(s). If None, the initial state is randomly
            drawn.

        num_reps : scalar(int), optional(default=None)
            Number of repetitions of simulation.

        random_state : int or np.random.RandomState/Generator, optional
            Random seed (integer) or np.random.RandomState or Generator
            instance to set the initial state of the random number
            generator for reproducibility. If None, a randomly
            initialized RandomState is used.

        Returns
        -------
        X : ndarray(ndim=1 or 2)
            Array containing the state values of the sample path(s). See
            the `simulate` method for more information.

        """
        random_state = check_random_state(random_state)
        dim = 1  # Dimension of the returned array: 1 or 2

        msg_out_of_range = 'index {init} is out of the state space'

        try:
            k = len(init)  # init is an array
            dim = 2
            init_states = np.asarray(init, dtype=int)
            # Check init_states are in the state space
            if (init_states >= self.n).any() or (init_states < -self.n).any():
                idx = np.where(
                    (init_states >= self.n) + (init_states < -self.n)
                )[0][0]
                raise ValueError(msg_out_of_range.format(init=idx))
            if num_reps is not None:
                k *= num_reps
                init_states = np.tile(init_states, num_reps)
        except TypeError:  # init is a scalar(int) or None
            k = 1
            if num_reps is not None:
                dim = 2
                k = num_reps
            if init is None:
                init_states = rng_integers(random_state, self.n, size=k)
            elif isinstance(init, numbers.Integral):
                # Check init is in the state space
                if init >= self.n or init < -self.n:
                    raise ValueError(msg_out_of_range.format(init=init))
                init_states = np.ones(k, dtype=int) * init
            else:
                raise ValueError(
                    'init must be int, array_like of ints, or None'
                )

        # === set up array to store output === #
        X = np.empty((k, ts_length), dtype=int)

        # Random values, uniformly sampled from [0, 1)
        random_values = random_state.random(size=(k, ts_length-1))

        # Generate sample paths and store in X
        if not self.is_sparse:  # Dense
            _generate_sample_paths(
                self.cdfs, init_states, random_values, out=X
            )
        else:  # Sparse
            _generate_sample_paths_sparse(
                self.cdfs1d, self.P.indices, self.P.indptr, init_states,
                random_values, out=X
            )

        if dim == 1:
            return X[0]
        else:
            return X

    def simulate(self, ts_length, init=None, num_reps=None, random_state=None):
        """
        Simulate time series of state transitions, where the states are
        annotated with their values (if `state_values` is not None).

        Parameters
        ----------
        ts_length : scalar(int)
            Length of each simulation.

        init : scalar or array_like, optional(default=None)
            Initial state values(s). If None, the initial state is
            randomly drawn.

        num_reps : scalar(int), optional(default=None)
            Number of repetitions of simulation.

        random_state : int or np.random.RandomState/Generator, optional
            Random seed (integer) or np.random.RandomState or Generator
            instance to set the initial state of the random number
            generator for reproducibility. If None, a randomly
            initialized RandomState is used.

        Returns
        -------
        X : ndarray(ndim=1 or 2)
            Array containing the sample path(s), of shape (ts_length,)
            if init is a scalar (integer) or None and num_reps is None;
            of shape (k, ts_length) otherwise, where k = len(init) if
            (init, num_reps) = (array, None), k = num_reps if (init,
            num_reps) = (int or None, int), and k = len(init)*num_reps
            if (init, num_reps) = (array, int).

        """
        if init is not None:
            init_idx = self.get_index(init)
        else:
            init_idx = None
        X = self.simulate_indices(ts_length, init=init_idx, num_reps=num_reps,
                                  random_state=random_state)

        # Annotate states
        if self.state_values is not None:
            X = self.state_values[X]

        return X


@jit(nopython=True)
def _generate_sample_paths(P_cdfs, init_states, random_values, out):
    """
    Generate num_reps sample paths of length ts_length, where num_reps =
    out.shape[0] and ts_length = out.shape[1].

    Parameters
    ----------
    P_cdfs : ndarray(float, ndim=2)
        Array containing as rows the CDFs of the state transition.

    init_states : array_like(int, ndim=1)
        Array containing the initial states. Its length must be equal to
        num_reps.

    random_values : ndarray(float, ndim=2)
        Array containing random values from [0, 1). Its shape must be
        equal to (num_reps, ts_length-1)

    out : ndarray(int, ndim=2)
        Array to store the sample paths.

    Notes
    -----
    This routine is jit-complied by Numba.

    """
    num_reps, ts_length = out.shape

    for i in range(num_reps):
        out[i, 0] = init_states[i]
        for t in range(ts_length-1):
            out[i, t+1] = searchsorted(P_cdfs[out[i, t]], random_values[i, t])


@jit(nopython=True)
def _generate_sample_paths_sparse(P_cdfs1d, indices, indptr, init_states,
                                  random_values, out):
    """
    For sparse matrix.

    Generate num_reps sample paths of length ts_length, where num_reps =
    out.shape[0] and ts_length = out.shape[1].

    Parameters
    ----------
    P_cdfs1d : ndarray(float, ndim=1)
        1D array containing the CDFs of the state transition.

    indices : ndarray(int, ndim=1)
        CSR format index array.

    indptr : ndarray(int, ndim=1)
        CSR format index pointer array.

    init_states : array_like(int, ndim=1)
        Array containing the initial states. Its length must be equal to
        num_reps.

    random_values : ndarray(float, ndim=2)
        Array containing random values from [0, 1). Its shape must be
        equal to (num_reps, ts_length-1)

    out : ndarray(int, ndim=2)
        Array to store the sample paths.

    Notes
    -----
    This routine is jit-complied by Numba.

    """
    num_reps, ts_length = out.shape

    for i in range(num_reps):
        out[i, 0] = init_states[i]
        for t in range(ts_length-1):
            k = searchsorted(P_cdfs1d[indptr[out[i, t]]:indptr[out[i, t]+1]],
                             random_values[i, t])
            out[i, t+1] = indices[indptr[out[i, t]]+k]


def mc_compute_stationary(P):
    """
    Computes stationary distributions of P, one for each recurrent
    class. Any stationary distribution is written as a convex
    combination of these distributions.

    Returns
    -------
    stationary_dists : array_like(float, ndim=2)
        Array containing the stationary distributions as its rows.

    """
    return MarkovChain(P).stationary_distributions


def mc_sample_path(P, init=0, sample_size=1000, random_state=None):
    """
    Generates one sample path from the Markov chain represented by
    (n x n) transition matrix P on state space S = {{0,...,n-1}}.

    Parameters
    ----------
    P : array_like(float, ndim=2)
        A Markov transition matrix.

    init : array_like(float ndim=1) or scalar(int), optional(default=0)
        If init is an array_like, then it is treated as the initial
        distribution across states.  If init is a scalar, then it
        treated as the deterministic initial state.

    sample_size : scalar(int), optional(default=1000)
        The length of the sample path.

    random_state : int or np.random.RandomState/Generator, optional
        Random seed (integer) or np.random.RandomState or Generator
        instance to set the initial state of the random number generator
        for reproducibility. If None, a randomly initialized RandomState
        is used.

    Returns
    -------
    X : array_like(int, ndim=1)
        The simulation of states.

    """
    random_state = check_random_state(random_state)

    if isinstance(init, numbers.Integral):
        X_0 = init
    else:
        cdf0 = np.cumsum(init)
        u_0 = random_state.random()
        X_0 = searchsorted(cdf0, u_0)

    mc = MarkovChain(P)
    return mc.simulate(ts_length=sample_size, init=X_0,
                       random_state=random_state)
