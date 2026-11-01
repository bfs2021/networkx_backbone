"""Tests for structural backbone extraction methods."""

import networkx as nx
import pytest

from networkx_backbone import (
    boolean_filter,
    doubly_stochastic_filter,
    edge_betweenness_filter,
    global_sparsification,
    global_threshold_filter,
    h_backbone,
    high_salience_skeleton,
    maximum_spanning_tree_backbone,
    metric_backbone,
    modularity_backbone,
    node_degree_filter,
    planar_maximally_filtered_graph,
    primary_linkage_analysis,
    strongest_n_ties,
    ultrametric_backbone,
)


class TestGlobalThresholdFilter:
    def test_scores_edges_by_weight_threshold(self, weighted_triangle):
        H = global_threshold_filter(weighted_triangle, threshold=2.0)
        assert H.number_of_edges() == weighted_triangle.number_of_edges()
        assert H[0][1]["global_threshold_keep"] is True
        assert H[0][2]["global_threshold_keep"] is True
        assert H[1][2]["global_threshold_keep"] is False

    def test_boolean_filter_applies_threshold(self, weighted_triangle):
        H = global_threshold_filter(weighted_triangle, threshold=10.0)
        backbone = boolean_filter(H, "global_threshold_keep")
        assert set(backbone.nodes()) == set(weighted_triangle.nodes())
        assert backbone.number_of_edges() == 0


class TestStrongestNTies:
    def test_scores_keep_flags(self, weighted_triangle):
        H = strongest_n_ties(weighted_triangle, n=1)
        assert H.number_of_edges() == weighted_triangle.number_of_edges()
        backbone = boolean_filter(H, "strongest_n_ties_keep")
        assert backbone.number_of_edges() == 2

    def test_n_less_than_1_raises(self, weighted_triangle):
        with pytest.raises(ValueError):
            strongest_n_ties(weighted_triangle, n=0)

    def test_preserves_all_nodes(self, weighted_triangle):
        H = strongest_n_ties(weighted_triangle, n=1)
        assert set(H.nodes()) == set(weighted_triangle.nodes())


class TestGlobalSparsification:
    def test_scores_fraction_of_edges(self, weighted_triangle):
        H = global_sparsification(weighted_triangle, s=0.5)
        assert H.number_of_edges() == weighted_triangle.number_of_edges()
        backbone = boolean_filter(H, "global_sparsification_keep")
        assert backbone.number_of_edges() == 2
        assert set(H.nodes()) == set(weighted_triangle.nodes())

    def test_invalid_fraction_raises(self, weighted_triangle):
        with pytest.raises(ValueError):
            global_sparsification(weighted_triangle, s=0.0)


class TestPrimaryLinkageAnalysis:
    def test_adds_keep_attribute(self, weighted_triangle):
        H = primary_linkage_analysis(weighted_triangle)
        assert isinstance(H, nx.Graph)
        assert set(H.nodes()) == set(weighted_triangle.nodes())
        for _, _, data in H.edges(data=True):
            assert "primary_linkage_keep" in data


class TestEdgeBetweennessFilter:
    def test_scores_fraction_and_adds_attribute(self, weighted_triangle):
        H = edge_betweenness_filter(weighted_triangle, s=0.5)
        assert H.number_of_edges() == weighted_triangle.number_of_edges()
        for _, _, data in H.edges(data=True):
            assert "edge_betweenness" in data
            assert "edge_betweenness_keep" in data
        backbone = boolean_filter(H, "edge_betweenness_keep")
        assert backbone.number_of_edges() == 2

    def test_invalid_fraction_raises(self, weighted_triangle):
        with pytest.raises(ValueError):
            edge_betweenness_filter(weighted_triangle, s=1.5)


class TestNodeDegreeFilter:
    def test_scores_nodes_and_edges_by_degree(self):
        G = nx.path_graph(4)
        H = node_degree_filter(G, min_degree=2)
        assert H.number_of_edges() == G.number_of_edges()
        assert H.nodes[1]["node_degree_keep"] is True
        assert H.nodes[0]["node_degree_keep"] is False
        backbone = boolean_filter(H, "node_degree_keep")
        assert backbone.number_of_edges() == 1

    def test_negative_degree_raises(self, karate):
        with pytest.raises(ValueError):
            node_degree_filter(karate, min_degree=-1)


class TestHighSalienceSkeleton:
    def test_adds_salience_attribute(self, weighted_triangle):
        H = high_salience_skeleton(weighted_triangle)
        for _, _, data in H.edges(data=True):
            assert "salience" in data
            assert 0.0 <= data["salience"] <= 1.0

    def test_preserves_edges(self, weighted_triangle):
        H = high_salience_skeleton(weighted_triangle)
        assert H.number_of_edges() == weighted_triangle.number_of_edges()

    def test_rejects_directed(self):
        G = nx.DiGraph()
        G.add_weighted_edges_from([(0, 1, 1.0)])
        with pytest.raises(nx.NetworkXNotImplemented):
            high_salience_skeleton(G)


class TestMetricBackbone:
    def test_adds_metric_keep_attribute(self, weighted_triangle):
        H = metric_backbone(weighted_triangle)
        assert H.number_of_edges() == weighted_triangle.number_of_edges()
        for _, _, data in H.edges(data=True):
            assert "metric_keep" in data

    def test_preserves_nodes(self, weighted_triangle):
        H = metric_backbone(weighted_triangle)
        assert set(H.nodes()) == set(weighted_triangle.nodes())


class TestUltrametricBackbone:
    def test_adds_ultrametric_keep_attribute(self, weighted_triangle):
        H = ultrametric_backbone(weighted_triangle)
        assert H.number_of_edges() == weighted_triangle.number_of_edges()
        for _, _, data in H.edges(data=True):
            assert "ultrametric_keep" in data


class TestDoublyStochasticFilter:
    def test_adds_ds_weight(self, weighted_triangle):
        H = doubly_stochastic_filter(weighted_triangle)
        for _, _, data in H.edges(data=True):
            assert "ds_weight" in data


class TestHBackbone:
    def test_adds_h_keep_attribute(self, weighted_triangle):
        H = h_backbone(weighted_triangle)
        assert H.number_of_edges() == weighted_triangle.number_of_edges()
        for _, _, data in H.edges(data=True):
            assert "h_backbone_keep" in data


class TestModularityBackbone:
    def test_adds_vitality(self, weighted_triangle):
        H = modularity_backbone(weighted_triangle)
        for node in H.nodes():
            assert "vitality" in H.nodes[node]


class TestPlanarMaximallyFilteredGraph:
    def test_result_is_planar(self, weighted_triangle):
        H = planar_maximally_filtered_graph(weighted_triangle)
        backbone = boolean_filter(H, "pmfg_keep")
        is_planar, _ = nx.check_planarity(backbone)
        assert is_planar

    def test_subset_of_original(self):
        G = nx.complete_graph(10)
        for u, v in G.edges():
            G[u][v]["weight"] = 1.0
        H = planar_maximally_filtered_graph(G)
        assert H.number_of_edges() == G.number_of_edges()
        backbone = boolean_filter(H, "pmfg_keep")
        assert backbone.number_of_edges() <= G.number_of_edges()


class TestMaximumSpanningTreeBackbone:
    def test_is_spanning_tree(self, weighted_triangle):
        H = maximum_spanning_tree_backbone(weighted_triangle)
        assert H.number_of_edges() == weighted_triangle.number_of_edges()
        backbone = boolean_filter(H, "mst_keep")
        assert backbone.number_of_edges() == weighted_triangle.number_of_nodes() - 1
        assert nx.is_connected(backbone)

    def test_selects_heaviest_edges(self):
        G = nx.Graph()
        G.add_weighted_edges_from([(0, 1, 5.0), (1, 2, 3.0), (0, 2, 1.0)])
        H = maximum_spanning_tree_backbone(G)
        backbone = boolean_filter(H, "mst_keep")
        assert backbone.has_edge(0, 1)  # weight 5.0
        assert backbone.has_edge(1, 2)  # weight 3.0
