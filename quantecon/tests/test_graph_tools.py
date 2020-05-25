"""
Tests for graph_tools.py

"""
import sys
import numpy as np
from numpy.testing import assert_array_equal, assert_raises
import nose
from nose.tools import eq_, ok_, raises

from quantecon.graph_tools import DiGraph, random_tournament_graph


def list_of_array_equal(s, t):
    """
    Compare two lists of ndarrays

    s, t: lists of numpy.ndarrays

    """
    eq_(len(s), len(t))
    all(assert_array_equal(x, y) for x, y in zip(s, t))


class Graphs:
    """Setup graphs for the tests"""

    def __init__(self):
        self.strongly_connected_graph_dicts = []
        self.not_strongly_connected_graph_dicts = []

        graph_dict = {
            'A': np.array([[1, 0], [0, 1]]),
            'strongly_connected_components':
            [np.array([0]), np.array([1])],
            'sink_strongly_connected_components':
            [np.array([0]), np.array([1])],
            'is_strongly_connected': False,
        }
        self.not_strongly_connected_graph_dicts.append(graph_dict)

        graph_dict = {
            'A': np.array([[1, 0, 0], [1, 0, 1], [0, 0, 1]]),
            'strongly_connected_components':
            [np.array([0]), np.array([1]), np.array([2])],
            'sink_strongly_connected_components':
            [np.array([0]), np.array([2])],
            'is_strongly_connected': False,
        }
        self.not_strongly_connected_graph_dicts.append(graph_dict)

        graph_dict = {
            'A': np.array([[1, 1], [1, 1]]),
            'strongly_connected_components': [np.arange(2)],
            'sink_strongly_connected_components': [np.arange(2)],
            'is_strongly_connected': True,
            'period': 1,
            'is_aperiodic': True,
            'cyclic_components': [np.arange(2)],
        }
        self.strongly_connected_graph_dicts.append(graph_dict)

        graph_dict = {
            'A': np.array([[0, 1], [1, 0]]),
            'strongly_connected_components': [np.arange(2)],
            'sink_strongly_connected_components': [np.arange(2)],
            'is_strongly_connected': True,
            'period': 2,
            'is_aperiodic': False,
            'cyclic_components': [np.array([0]), np.array([1])],
        }
        self.strongly_connected_graph_dicts.append(graph_dict)

        graph_dict = {
            'A': np.array([[0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [1, 0, 0, 1],
                           [0, 0, 1, 0]]),
            'strongly_connected_components': [np.arange(4)],
            'sink_strongly_connected_components': [np.arange(4)],
            'is_strongly_connected': True,
            'period': 1,
            'is_aperiodic': True,
            'cyclic_components': [np.arange(4)],
        }
        self.strongly_connected_graph_dicts.append(graph_dict)

        # Weighted graph
        graph_dict = {
            'A': np.array([[0, 0.5], [2, 0]]),
            'weighted': True,
            'strongly_connected_components': [np.arange(2)],
            'sink_strongly_connected_components': [np.arange(2)],
            'is_strongly_connected': True,
            'period': 2,
            'is_aperiodic': False,
            'cyclic_components': [np.array([0]), np.array([1])],
        }
        self.strongly_connected_graph_dicts.append(graph_dict)

        # Degenrate graph with no edge
        graph_dict = {
            'A': np.array([[0]]),
            'strongly_connected_components': [np.arange(1)],
            'sink_strongly_connected_components': [np.arange(1)],
            'is_strongly_connected': True,
            'period': 1,
            'is_aperiodic': True,
            'cyclic_components': [np.array([0])],
        }
        self.strongly_connected_graph_dicts.append(graph_dict)

        # Degenrate graph with self loop
        graph_dict = {
            'A': np.array([[1]]),
            'strongly_connected_components': [np.arange(1)],
            'sink_strongly_connected_components': [np.arange(1)],
            'is_strongly_connected': True,
            'period': 1,
            'is_aperiodic': True,
            'cyclic_components': [np.array([0])],
        }
        self.strongly_connected_graph_dicts.append(graph_dict)

        self.graph_dicts = \
            self.strongly_connected_graph_dicts + \
            self.not_strongly_connected_graph_dicts


class TestDiGraph:
    """Test the methods in Digraph"""

    def setUp(self):
        """Setup Digraph instances"""
        self.graphs = Graphs()
        for graph_dict in self.graphs.graph_dicts:
            try:
                weighted = graph_dict['weighted']
            except:
                weighted = False
            graph_dict['g'] = DiGraph(graph_dict['A'], weighted=weighted)

    def test_strongly_connected_components(self):
        for graph_dict in self.graphs.graph_dicts:
            list_of_array_equal(
                sorted(graph_dict['g'].strongly_connected_components,
                       key=lambda x: x[0]),
                sorted(graph_dict['strongly_connected_components'],
                       key=lambda x: x[0])
            )

    def test_num_strongly_connected_components(self):
        for graph_dict in self.graphs.graph_dicts:
            eq_(graph_dict['g'].num_strongly_connected_components,
                len(graph_dict['strongly_connected_components']))

    def test_sink_strongly_connected_components(self):
        for graph_dict in self.graphs.graph_dicts:
            list_of_array_equal(
                sorted(graph_dict['g'].sink_strongly_connected_components,
                       key=lambda x: x[0]),
                sorted(graph_dict['sink_strongly_connected_components'],
                       key=lambda x: x[0])
            )

    def test_num_sink_strongly_connected_components(self):
        for graph_dict in self.graphs.graph_dicts:
            eq_(graph_dict['g'].num_sink_strongly_connected_components,
                len(graph_dict['sink_strongly_connected_components']))

    def test_is_strongly_connected(self):
        for graph_dict in self.graphs.graph_dicts:
            eq_(graph_dict['g'].is_strongly_connected,
                graph_dict['is_strongly_connected'])

    def test_period(self):
        for graph_dict in self.graphs.graph_dicts:
            try:
                eq_(graph_dict['g'].period, graph_dict['period'])
            except NotImplementedError:
                eq_(graph_dict['g'].is_strongly_connected, False)

    def test_is_aperiodic(self):
        for graph_dict in self.graphs.graph_dicts:
            try:
                eq_(graph_dict['g'].is_aperiodic,
                    graph_dict['is_aperiodic'])
            except NotImplementedError:
                eq_(graph_dict['g'].is_strongly_connected, False)

    def test_cyclic_components(self):
        for graph_dict in self.graphs.graph_dicts:
            try:
                list_of_array_equal(
                    sorted(graph_dict['g'].cyclic_components,
                           key=lambda x: x[0]),
                    sorted(graph_dict['cyclic_components'],
                           key=lambda x: x[0])
                )
            except NotImplementedError:
                eq_(graph_dict['g'].is_strongly_connected, False)


def test_subgraph():
    adj_matrix = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
    g = DiGraph(adj_matrix)
    nodes = [1, 2]

    subgraph_adj_matrix = [[False, True], [False, False]]
    assert_array_equal(
        g.subgraph(nodes).csgraph.toarray(),
        subgraph_adj_matrix
    )


def test_subgraph_weighted():
    adj_matrix = np.arange(3**2).reshape(3, 3)
    g = DiGraph(adj_matrix, weighted=True)
    nodes = [0, 1]

    subgraph_adj_matrix = [[0, 1], [3, 4]]
    assert_array_equal(
        g.subgraph(nodes).csgraph.toarray(),
        subgraph_adj_matrix
    )


def test_node_labels_connected_components():
    adj_matrix = [[1, 0, 0], [1, 0, 0], [0, 0, 1]]
    node_labels = np.array(['a', 'b', 'c'])
    g = DiGraph(adj_matrix, node_labels=node_labels)

    sccs = [[0], [1], [2]]
    sink_sccs = [[0], [2]]

    properties = ['strongly_connected_components',
                  'sink_strongly_connected_components']
    suffix = '_indices'
    for prop0, components_ind in zip(properties, [sccs, sink_sccs]):
        for return_indices in [True, False]:
            if return_indices:
                components = components_ind
                prop = prop0 + suffix
            else:
                components = [node_labels[i] for i in components_ind]
                prop = prop0
            list_of_array_equal(
                sorted(getattr(g, prop), key=lambda x: x[0]),
                sorted(components, key=lambda x: x[0])
            )


def test_node_labels_cyclic_components():
    adj_matrix = [[0, 1], [1, 0]]
    node_labels = np.array(['a', 'b'])
    g = DiGraph(adj_matrix, node_labels=node_labels)

    cyclic_components = [[0], [1]]

    properties = ['cyclic_components']
    suffix = '_indices'
    for prop0, components_ind in zip(properties, [cyclic_components]):
        for return_indices in [True, False]:
            if return_indices:
                components = components_ind
                prop = prop0 + suffix
            else:
                components = [node_labels[i] for i in components_ind]
                prop = prop0
            list_of_array_equal(
                sorted(getattr(g, prop), key=lambda x: x[0]),
                sorted(components, key=lambda x: x[0])
            )


def test_node_labels_subgraph():
    adj_matrix = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
    node_labels = np.array(['a', 'b', 'c'])
    g = DiGraph(adj_matrix, node_labels=node_labels)
    nodes = [1, 2]

    assert_array_equal(
        g.subgraph(nodes).node_labels,
        node_labels[nodes]
    )


@raises(ValueError)
def test_raises_value_error_non_sym():
    """Test with non symmetric input"""
    DiGraph(np.array([[0.4, 0.6]]))


def test_raises_non_homogeneous_node_labels():
    adj_matrix = [[1, 0], [0, 1]]
    node_labels = [(0, 1), 2]
    assert_raises(ValueError, DiGraph, adj_matrix, node_labels=node_labels)


class TestRandomTournamentGraph:
    def setUp(self):
        n = 5
        g = random_tournament_graph(n)
        self.adj_matrix = g.csgraph.toarray()
        self.eye_bool = np.eye(n, dtype=bool)

    def test_diagonal(self):
        # Test no self loop
        ok_(not self.adj_matrix[self.eye_bool].any())

    def test_off_diagonal(self):
        # Test for each pair of distinct nodes to have exactly one edge
        ok_((self.adj_matrix ^ self.adj_matrix.T)[~self.eye_bool].all())


def test_random_tournament_graph_seed():
    n = 7
    seed = 1234
    graphs = [random_tournament_graph(n, random_state=seed) for i in range(2)]
    assert_array_equal(*[g.csgraph.toarray() for g in graphs])


if __name__ == '__main__':
    argv = sys.argv[:]
    argv.append('--verbose')
    argv.append('--nocapture')
    nose.main(argv=argv, defaultTest=__file__)
