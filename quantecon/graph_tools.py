"""
Tools for dealing with a directed graph.

"""
import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from fractions import gcd
from numba import jit

from .util import check_random_state


# Decorator for *_components properties
def annotate_nodes(func):
    def new_func(self):
        list_of_components = func(self)
        if self.node_labels is not None:
            return [self.node_labels[c] for c in list_of_components]
        return list_of_components
    return new_func


class DiGraph:
    r"""
    Class for a directed graph. It stores useful information about the
    graph structure such as strong connectivity [1]_ and periodicity
    [2]_.

    Parameters
    ----------
    adj_matrix : array_like(ndim=2)
        Adjacency matrix representing a directed graph. Must be of shape
        n x n.

    weighted : bool, optional(default=False)
        Whether to treat `adj_matrix` as a weighted adjacency matrix.

    node_labels : array_like(default=None)
        Array_like of length n containing the labels associated with the
        nodes, which must be homogeneous in type. If None, the labels
        default to integers 0 through n-1.

    Attributes
    ----------
    csgraph : scipy.sparse.csr_matrix
        Compressed sparse representation of the digraph.

    is_strongly_connected : bool
        Indicate whether the digraph is strongly connected.

    num_strongly_connected_components : int
        The number of the strongly connected components.

    strongly_connected_components_indices : list(ndarray(int))
        List of numpy arrays containing the indices of the strongly
        connected components.

    strongly_connected_components : list(ndarray)
        List of numpy arrays containing the strongly connected
        components, where the nodes are annotated with their labels (if
        `node_labels` is not None).

    num_sink_strongly_connected_components : int
        The number of the sink strongly connected components.

    sink_strongly_connected_components_indices : list(ndarray(int))
        List of numpy arrays containing the indices of the sink strongly
        connected components.

    sink_strongly_connected_components : list(ndarray)
        List of numpy arrays containing the sink strongly connected
        components, where the nodes are annotated with their labels (if
        `node_labels` is not None).

    is_aperiodic : bool
        Indicate whether the digraph is aperiodic.

    period : int
        The period of the digraph. Defined only for a strongly connected
        digraph.

    cyclic_components_indices : list(ndarray(int))
        List of numpy arrays containing the indices of the cyclic
        components.

    cyclic_components : list(ndarray)
        List of numpy arrays containing the cyclic components, where the
        nodes are annotated with their labels (if `node_labels` is not
        None).

    References
    ----------
    .. [1] `Strongly connected component
       <http://en.wikipedia.org/wiki/Strongly_connected_component>`_,
       Wikipedia.

    .. [2] `Aperiodic graph
       <http://en.wikipedia.org/wiki/Aperiodic_graph>`_, Wikipedia.

    """

    def __init__(self, adj_matrix, weighted=False, node_labels=None):
        if weighted:
            dtype = None
        else:
            dtype = bool
        self.csgraph = sparse.csr_matrix(adj_matrix, dtype=dtype)

        m, n = self.csgraph.shape
        if n != m:
            raise ValueError('input matrix must be square')

        self.n = n  # Number of nodes

        # Call the setter method
        self.node_labels = node_labels

        self._num_scc = None
        self._scc_proj = None
        self._sink_scc_labels = None

        self._period = None

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Directed Graph:\n  - n(number of nodes): {n}".format(n=self.n)

    @property
    def node_labels(self):
        return self._node_labels

    @node_labels.setter
    def node_labels(self, values):
        if values is None:
            self._node_labels = None
        else:
            values = np.asarray(values)
            if (values.ndim < 1) or (values.shape[0] != self.n):
                raise ValueError(
                    'node_labels must be an array_like of length n'
                )
            if np.issubdtype(values.dtype, np.object_):
                raise ValueError(
                    'data in node_labels must be homogeneous in type'
                )
            self._node_labels = values

    def _find_scc(self):
        """
        Set ``self._num_scc`` and ``self._scc_proj``
        by calling ``scipy.sparse.csgraph.connected_components``:
        * docs.scipy.org/doc/scipy/reference/sparse.csgraph.html
        * github.com/scipy/scipy/blob/master/scipy/sparse/csgraph/_traversal.pyx

        ``self._scc_proj`` is a list of length `n` that assigns to each node
        the label of the strongly connected component to which it belongs.

        """
        # Find the strongly connected components
        self._num_scc, self._scc_proj = \
            csgraph.connected_components(self.csgraph, connection='strong')

    @property
    def num_strongly_connected_components(self):
        if self._num_scc is None:
            self._find_scc()
        return self._num_scc

    @property
    def scc_proj(self):
        if self._scc_proj is None:
            self._find_scc()
        return self._scc_proj

    @property
    def is_strongly_connected(self):
        return (self.num_strongly_connected_components == 1)

    def _condensation_lil(self):
        """
        Return the sparse matrix representation of the condensation digraph
        in lil format.

        """
        condensation_lil = sparse.lil_matrix(
            (self.num_strongly_connected_components,
             self.num_strongly_connected_components), dtype=bool
        )

        scc_proj = self.scc_proj
        for node_from, node_to in _csr_matrix_indices(self.csgraph):
            scc_from, scc_to = scc_proj[node_from], scc_proj[node_to]
            if scc_from != scc_to:
                condensation_lil[scc_from, scc_to] = True

        return condensation_lil

    def _find_sink_scc(self):
        """
        Set self._sink_scc_labels, which is a list containing the labels of
        the strongly connected components.

        """
        condensation_lil = self._condensation_lil()

        # A sink SCC is a SCC such that none of its members is strongly
        # connected to nodes in other SCCs
        # Those k's such that graph_condensed_lil.rows[k] == []
        self._sink_scc_labels = \
            np.where(np.logical_not(condensation_lil.rows))[0]

    @property
    def sink_scc_labels(self):
        if self._sink_scc_labels is None:
            self._find_sink_scc()
        return self._sink_scc_labels

    @property
    def num_sink_strongly_connected_components(self):
        return len(self.sink_scc_labels)

    @property
    def strongly_connected_components_indices(self):
        if self.is_strongly_connected:
            return [np.arange(self.n)]
        else:
            return [np.where(self.scc_proj == k)[0]
                    for k in range(self.num_strongly_connected_components)]

    @property
    @annotate_nodes
    def strongly_connected_components(self):
        return self.strongly_connected_components_indices

    @property
    def sink_strongly_connected_components_indices(self):
        if self.is_strongly_connected:
            return [np.arange(self.n)]
        else:
            return [np.where(self.scc_proj == k)[0]
                    for k in self.sink_scc_labels.tolist()]

    @property
    @annotate_nodes
    def sink_strongly_connected_components(self):
        return self.sink_strongly_connected_components_indices

    def _compute_period(self):
        """
        Set ``self._period`` and ``self._cyclic_components_proj``.

        Use the algorithm described in:
        J. P. Jarvis and D. R. Shier,
        "Graph-Theoretic Analysis of Finite Markov Chains," 1996.

        """
        # Degenerate graph with a single node (which is strongly connected)
        # csgraph.reconstruct_path would raise an exception
        # github.com/scipy/scipy/issues/4018
        if self.n == 1:
            if self.csgraph[0, 0] == 0:  # No edge: "trivial graph"
                self._period = 1  # Any universally accepted definition?
                self._cyclic_components_proj = np.zeros(self.n, dtype=int)
                return None
            else:  # Self loop
                self._period = 1
                self._cyclic_components_proj = np.zeros(self.n, dtype=int)
                return None

        if not self.is_strongly_connected:
            raise NotImplementedError(
                'Not defined for a non strongly-connected digraph'
            )

        if np.any(self.csgraph.diagonal() > 0):
            self._period = 1
            self._cyclic_components_proj = np.zeros(self.n, dtype=int)
            return None

        # Construct a breadth-first search tree rooted at 0
        node_order, predecessors = \
            csgraph.breadth_first_order(self.csgraph, i_start=0)
        bfs_tree_csr = \
            csgraph.reconstruct_path(self.csgraph, predecessors)

        # Edges not belonging to tree_csr
        non_bfs_tree_csr = self.csgraph - bfs_tree_csr
        non_bfs_tree_csr.eliminate_zeros()

        # Distance to 0
        level = np.zeros(self.n, dtype=int)
        for i in range(1, self.n):
            level[node_order[i]] = level[predecessors[node_order[i]]] + 1

        # Determine the period
        d = 0
        for node_from, node_to in _csr_matrix_indices(non_bfs_tree_csr):
            value = level[node_from] - level[node_to] + 1
            d = gcd(d, value)
            if d == 1:
                self._period = 1
                self._cyclic_components_proj = np.zeros(self.n, dtype=int)
                return None

        self._period = d
        self._cyclic_components_proj = level % d

    @property
    def period(self):
        if self._period is None:
            self._compute_period()
        return self._period

    @property
    def is_aperiodic(self):
        return (self.period == 1)

    @property
    def cyclic_components_indices(self):
        if self.is_aperiodic:
            return [np.arange(self.n)]
        else:
            return [np.where(self._cyclic_components_proj == k)[0]
                    for k in range(self.period)]

    @property
    @annotate_nodes
    def cyclic_components(self,):
        return self.cyclic_components_indices

    def subgraph(self, nodes):
        """
        Return the subgraph consisting of the given nodes and edges
        between thses nodes.

        Parameters
        ----------
        nodes : array_like(int, ndim=1)
           Array of node indices.

        Returns
        -------
        DiGraph
            A DiGraph representing the subgraph.

        """
        adj_matrix = self.csgraph[np.ix_(nodes, nodes)]

        weighted = True  # To copy the dtype

        if self.node_labels is not None:
            node_labels = self.node_labels[nodes]
        else:
            node_labels = None

        return DiGraph(adj_matrix, weighted=weighted, node_labels=node_labels)


def _csr_matrix_indices(S):
    """
    Generate the indices of nonzero entries of a csr_matrix S

    """
    m, n = S.shape

    for i in range(m):
        for j in range(S.indptr[i], S.indptr[i+1]):
            row_index, col_index = i, S.indices[j]
            yield row_index, col_index


def random_tournament_graph(n, random_state=None):
    """
    Return a random tournament graph [1]_ with n nodes.

    Parameters
    ----------
    n : scalar(int)
        Number of nodes.

    random_state : int or np.random.RandomState, optional
        Random seed (integer) or np.random.RandomState instance to set
        the initial state of the random number generator for
        reproducibility. If None, a randomly initialized RandomState is
        used.

    Returns
    -------
    DiGraph
        A DiGraph representing the tournament graph.

    References
    ----------
    .. [1] `Tournament (graph theory)
       <https://en.wikipedia.org/wiki/Tournament_(graph_theory)>`_,
       Wikipedia.

    """
    random_state = check_random_state(random_state)
    num_edges = n * (n-1) // 2
    r = random_state.random_sample(num_edges)
    row = np.empty(num_edges, dtype=int)
    col = np.empty(num_edges, dtype=int)
    _populate_random_tournament_row_col(n, r, row, col)
    data = np.ones(num_edges, dtype=bool)
    adj_matrix = sparse.coo_matrix((data, (row, col)), shape=(n, n))
    return DiGraph(adj_matrix)


@jit(nopython=True, cache=True)
def _populate_random_tournament_row_col(n, r, row, col):
    """
    Populate ndarrays `row` and `col` with directed edge indices
    determined by random numbers in `r` for a tournament graph with n
    nodes, which has num_edges = n * (n-1) // 2 edges.

    Parameters
    ----------
    n : scalar(int)
        Number of nodes.

    r : ndarray(float, ndim=1)
        ndarray of length num_edges containing random numbers in [0, 1).

    row, col : ndarray(int, ndim=1)
        ndarrays of length num_edges to be modified in place.

    """
    k = 0
    for i in range(n):
        for j in range(i+1, n):
            if r[k] < 0.5:
                row[k], col[k] = i, j
            else:
                row[k], col[k] = j, i
            k += 1
