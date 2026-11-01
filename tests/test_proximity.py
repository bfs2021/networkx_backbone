"""Known-answer and convention tests for networkx_backbone.proximity.

Shared behaviour (attributes, multigraph rejection, empty graphs, mixed labels,
no input mutation) is covered by ``test_api_contract.py``.
"""

import math

import networkx as nx
import pytest

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

NEIGHBOURHOOD_SCORES = [
    (neighborhood_overlap, "overlap"),
    (jaccard_backbone, "jaccard"),
    (dice_backbone, "dice"),
    (cosine_backbone, "cosine"),
    (hub_promoted_index, "hpi"),
    (hub_depressed_index, "hdi"),
    (lhn_local_index, "lhn_local"),
    (preferential_attachment_score, "pa"),
    (adamic_adar_index, "adamic_adar"),
    (resource_allocation_index, "resource_allocation"),
]


def test_complete_graph_known_values(complete4):
    # In K4 every pair has degree 3 and shares 2 common neighbours (degree 3).
    # Local path: (A^2)_{01} = 2 and (A^3)_{01} = 7, so 2 + 0.01 * 7.
    expected = {
        "overlap": 2,
        "jaccard": 2 / 4,
        "dice": 2 * 2 / 6,
        "cosine": 2 / 3,
        "hpi": 2 / 3,
        "hdi": 2 / 3,
        "lhn_local": 2 / 9,
        "pa": 9,
        "adamic_adar": 2 / math.log(3),
        "resource_allocation": 2 / 3,
        "lp": 2.07,
    }
    for func, attr in [*NEIGHBOURHOOD_SCORES, (local_path_index, "lp")]:
        H = func(complete4)
        for u, v, score in H.edges(data=attr):
            assert score == pytest.approx(expected[attr]), (attr, u, v)


def test_neighborhood_overlap_path_graph():
    # Nodes 1 and 2 share no common neighbours.
    assert neighborhood_overlap(nx.path_graph(4))[1][2]["overlap"] == 0


class TestGraphDistanceProximity:
    def test_existing_edges_only_by_default(self):
        G = nx.path_graph(4)
        H = graph_distance_proximity(G)
        assert H[0][1]["dist"] == 1.0
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


class TestSelfLoops:
    @pytest.fixture
    def looped(self):
        G = nx.Graph([(0, 1), (1, 2), (0, 2), (2, 3)])
        G.add_edge(0, 0)
        return G

    def test_self_loop_is_not_a_common_neighbour(self):
        G = nx.Graph([(0, 1)])
        G.add_edge(0, 0)
        assert neighborhood_overlap(G)[0][1]["overlap"] == 0
        assert jaccard_backbone(G)[0][1]["jaccard"] == 0.0
        assert resource_allocation_index(G)[0][1]["resource_allocation"] == 0.0

    def test_degrees_ignore_self_loops(self, looped):
        # Same scores as without the self-loop.
        plain = looped.copy()
        plain.remove_edge(0, 0)
        for func, attr in NEIGHBOURHOOD_SCORES:
            H, P = func(looped), func(plain)
            for u, v in plain.edges():
                assert H[u][v][attr] == P[u][v][attr], attr

    def test_local_path_index_ignores_self_loops(self, looped):
        plain = looped.copy()
        plain.remove_edge(0, 0)
        H, P = local_path_index(looped), local_path_index(plain)
        for u, v in plain.edges():
            assert H[u][v]["lp"] == pytest.approx(P[u][v]["lp"])

    def test_graph_distance_self_loop_zero(self, looped):
        H = graph_distance_proximity(looped)
        assert H[0][0]["dist"] == 0.0
        assert H[0][1]["dist"] == 1.0


class TestDirected:
    @pytest.fixture
    def D(self):
        # successors: 0 -> {1, 2}, 1 -> {2}, 2 -> {3}, 3 -> {0, 1}
        return nx.DiGraph([(0, 1), (0, 2), (1, 2), (2, 3), (3, 0), (3, 1)])

    def test_out_degree_denominators(self, D):
        # common successors of (0, 1): {2}; out-degrees k0 = 2, k1 = 1
        assert neighborhood_overlap(D)[0][1]["overlap"] == 1
        assert jaccard_backbone(D)[0][1]["jaccard"] == pytest.approx(1 / 2)
        assert dice_backbone(D)[0][1]["dice"] == pytest.approx(2 / 3)
        assert cosine_backbone(D)[0][1]["cosine"] == pytest.approx(1 / math.sqrt(2))
        assert hub_promoted_index(D)[0][1]["hpi"] == pytest.approx(1.0)
        assert hub_depressed_index(D)[0][1]["hdi"] == pytest.approx(0.5)
        assert lhn_local_index(D)[0][1]["lhn_local"] == pytest.approx(0.5)
        assert preferential_attachment_score(D)[0][1]["pa"] == 2
        # z = 2 has out-degree 1: RA adds 1/1; AA skips log(1) = 0
        assert resource_allocation_index(D)[0][1]["resource_allocation"] == 1.0
        assert adamic_adar_index(D)[0][1]["adamic_adar"] == 0.0

    def test_local_path_directed_walks(self, D):
        np = pytest.importorskip("numpy")
        pytest.importorskip("scipy")
        H = local_path_index(D, epsilon=0.5)
        A = nx.to_numpy_array(D, nodelist=list(D), weight=None)
        S = A @ A + 0.5 * (A @ A @ A)
        idx = {n: i for i, n in enumerate(D)}
        for u, v in D.edges():
            assert H[u][v]["lp"] == pytest.approx(S[idx[u], idx[v]])
        assert np.isfinite(H[0][1]["lp"])


class TestLocalPathWeight:
    def test_default_is_unweighted(self, weighted_triangle):
        pytest.importorskip("scipy")
        H = local_path_index(weighted_triangle, epsilon=0.1)
        # each edge: one 2-walk and three 3-walks in a triangle; weights ignored
        for u, v in H.edges():
            assert H[u][v]["lp"] == pytest.approx(1 + 0.1 * 3)

    def test_weighted_variant(self, weighted_triangle):
        pytest.importorskip("scipy")
        H = local_path_index(weighted_triangle, epsilon=0.0, weight="weight")
        # (A^2)_{01} = w02 * w21 = 2 * 1
        assert H[0][1]["lp"] == pytest.approx(2.0)
