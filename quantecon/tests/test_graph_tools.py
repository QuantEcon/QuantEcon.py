"""
Filename: test_graph_tools.py
Author: Daisuke Oyama

Tests for graph_tools.py

"""
import sys
import numpy as np
from numpy.testing import assert_array_equal
import nose
from nose.tools import eq_, ok_, raises

from quantecon.graph_tools import DiGraph


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


@raises(ValueError)
def test_raises_value_error_non_sym():
    """Test with non symmetric input"""
    g = DiGraph(np.array([[0.4, 0.6]]))


if __name__ == '__main__':
    argv = sys.argv[:]
    argv.append('--verbose')
    argv.append('--nocapture')
    nose.main(argv=argv, defaultTest=__file__)
