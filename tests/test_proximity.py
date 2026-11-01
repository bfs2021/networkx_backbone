"""Tests for proximity-based edge scoring methods."""


import networkx as nx

from networkx_backbone import (
    adamic_adar_index,
    cosine_backbone,
    dice_backbone,
    graph_distance_proximity,
    hub_depressed_index,
    hub_promoted_index,
    jaccard_backbone,
    lhn_local_index,
    local_path_index,
    neighborhood_overlap,
    preferential_attachment_score,
    resource_allocation_index,
)


class TestNeighborhoodOverlap:
    def test_complete_graph(self, complete4):
        H = neighborhood_overlap(complete4)
        # In K4, each pair shares 2 common neighbors
        assert H[0][1]["overlap"] == 2

    def test_path_graph(self):
        G = nx.path_graph(4)
        H = neighborhood_overlap(G)
        # Nodes 1 and 2 share 0 common neighbors
        assert H[1][2]["overlap"] == 0


class TestJaccardBackbone:
    def test_complete_graph(self, complete4):
        H = jaccard_backbone(complete4)
        # K4: 2 common / (3+3-2) = 2/4 = 0.5
        assert H[0][1]["jaccard"] == 0.5

    def test_values_in_range(self, complete4):
        H = jaccard_backbone(complete4)
        for u, v in H.edges():
            assert 0.0 <= H[u][v]["jaccard"] <= 1.0


class TestDiceBackbone:
    def test_complete_graph(self, complete4):
        H = dice_backbone(complete4)
        # K4: 2*2 / (3+3) = 4/6 ≈ 0.6667
        assert round(H[0][1]["dice"], 4) == 0.6667


class TestCosineBackbone:
    def test_complete_graph(self, complete4):
        H = cosine_backbone(complete4)
        # K4: 2 / sqrt(3*3) = 2/3 ≈ 0.6667
        assert round(H[0][1]["cosine"], 4) == 0.6667


class TestHubPromotedIndex:
    def test_complete_graph(self, complete4):
        H = hub_promoted_index(complete4)
        # K4: 2 / min(3,3) = 2/3
        assert round(H[0][1]["hpi"], 4) == 0.6667


class TestHubDepressedIndex:
    def test_complete_graph(self, complete4):
        H = hub_depressed_index(complete4)
        # K4: 2 / max(3,3) = 2/3
        assert round(H[0][1]["hdi"], 4) == 0.6667


class TestLHNLocalIndex:
    def test_complete_graph(self, complete4):
        H = lhn_local_index(complete4)
        # K4: 2 / (3*3) = 2/9 ≈ 0.2222
        assert round(H[0][1]["lhn_local"], 4) == 0.2222


class TestPreferentialAttachmentScore:
    def test_complete_graph(self, complete4):
        H = preferential_attachment_score(complete4)
        # K4: 3 * 3 = 9
        assert H[0][1]["pa"] == 9


class TestAdamicAdarIndex:
    def test_complete_graph(self, complete4):
        H = adamic_adar_index(complete4)
        # K4: 2 * 1/log(3) ≈ 1.8205
        assert round(H[0][1]["adamic_adar"], 4) == 1.8205


class TestResourceAllocationIndex:
    def test_complete_graph(self, complete4):
        H = resource_allocation_index(complete4)
        # K4: 2 * 1/3 ≈ 0.6667
        assert round(H[0][1]["resource_allocation"], 4) == 0.6667


class TestGraphDistanceProximity:
    def test_adjacent_nodes(self):
        G = nx.path_graph(4)
        H = graph_distance_proximity(G)
        assert H[0][1]["dist"] == 1.0

    def test_default_existing_edges_only(self):
        G = nx.path_graph(4)
        H = graph_distance_proximity(G)
        assert H.number_of_edges() == G.number_of_edges()
        assert not H.has_edge(0, 2)

    def test_all_pairs_mode(self):
        G = nx.path_graph(4)
        H = graph_distance_proximity(G, all_pairs=True)
        assert H.number_of_edges() == 6
        assert H[0][2]["dist"] == 0.5
        assert H[0][3]["dist"] == 1.0 / 3.0

    def test_all_pairs_mode_directed(self):
        G = nx.DiGraph([(0, 1), (1, 2)])
        H = graph_distance_proximity(G, all_pairs=True)
        assert H.number_of_edges() == 6
        assert H[0][2]["dist"] == 0.5
        assert H[2][0]["dist"] == 0.0


class TestLocalPathIndex:
    def test_positive_scores(self, complete4):
        H = local_path_index(complete4)
        for u, v in H.edges():
            assert H[u][v]["lp"] > 0
