"""Tests for evaluation measures."""

import networkx as nx

from networkx_backbone import (
    compare_backbones,
    edge_fraction,
    ks_degree,
    ks_weight,
    node_fraction,
    reachability,
    weight_fraction,
)


class TestNodeFraction:
    def test_identical_graphs(self):
        G = nx.path_graph(4)
        assert node_fraction(G, G) == 1.0

    def test_partial_backbone(self):
        G = nx.path_graph(4)
        H = nx.path_graph(3)
        assert node_fraction(G, H) == 0.75

    def test_empty_original(self):
        G = nx.Graph()
        H = nx.Graph()
        assert node_fraction(G, H) == 0.0


class TestEdgeFraction:
    def test_identical_graphs(self):
        G = nx.complete_graph(4)
        assert edge_fraction(G, G) == 1.0

    def test_half_edges(self):
        G = nx.complete_graph(4)  # 6 edges
        H = nx.path_graph(4)  # 3 edges
        assert edge_fraction(G, H) == 0.5


class TestWeightFraction:
    def test_half_weight(self):
        G = nx.Graph()
        G.add_weighted_edges_from([(0, 1, 3.0), (1, 2, 1.0), (0, 2, 2.0)])
        H = nx.Graph()
        H.add_weighted_edges_from([(0, 1, 3.0)])
        assert weight_fraction(G, H) == 0.5


class TestReachability:
    def test_connected_graph(self):
        G = nx.path_graph(4)
        assert reachability(G) == 1.0

    def test_isolated_nodes(self):
        G = nx.Graph()
        G.add_nodes_from([0, 1, 2])
        assert reachability(G) == 0.0

    def test_single_node(self):
        G = nx.Graph()
        G.add_node(0)
        assert reachability(G) == 1.0


class TestKSDegree:
    def test_identical_graphs(self):
        G = nx.complete_graph(5)
        assert ks_degree(G, G) == 0.0


class TestKSWeight:
    def test_identical_graphs(self):
        G = nx.Graph()
        G.add_weighted_edges_from([(0, 1, 3.0), (1, 2, 1.0)])
        assert ks_weight(G, G) == 0.0


class TestCompareBackbones:
    def test_returns_dict(self):
        G = nx.complete_graph(5)
        H = nx.path_graph(5)
        results = compare_backbones(G, {"path": H}, measures=[edge_fraction])
        assert "edge_fraction" in results["path"]
        assert isinstance(results["path"]["edge_fraction"], float)
