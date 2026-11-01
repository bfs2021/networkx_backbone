"""Regression tests for proximity-score conventions (self-loops, directed, multigraphs)."""

import math

import networkx as nx
import pytest

import networkx_backbone as nb
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

PROXIMITY = [
    "neighborhood_overlap",
    "jaccard_backbone",
    "dice_backbone",
    "cosine_backbone",
    "hub_promoted_index",
    "hub_depressed_index",
    "lhn_local_index",
    "preferential_attachment_score",
    "adamic_adar_index",
    "resource_allocation_index",
    "graph_distance_proximity",
    "local_path_index",
]


@pytest.mark.parametrize("name", PROXIMITY)
def test_multigraph_not_implemented(name):
    for M in (nx.MultiGraph([(0, 1), (0, 1), (1, 2)]), nx.MultiDiGraph([(0, 1)])):
        with pytest.raises(nx.NetworkXNotImplemented):
            getattr(nb, name)(M)


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
        for func, attr in [
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
        ]:
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

    def test_empty_graph(self):
        pytest.importorskip("scipy")
        assert local_path_index(nx.empty_graph(3)).number_of_edges() == 0


def test_mixed_type_nodes():
    G = nx.Graph([("a", 1), (1, 2.5), ("a", 2.5)])
    for name in PROXIMITY:
        H = getattr(nb, name)(G)
        assert H.number_of_edges() == 3
