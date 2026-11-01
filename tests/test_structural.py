"""Known-answer and regression tests for networkx_backbone.structural.

Shared behaviour (attributes, weight validation, directed/multigraph
rejection, empty graphs, mixed labels, no input mutation) is covered by
``test_api_contract.py``.
"""

import random
import warnings

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


def _kept(H, attr):
    return {frozenset((u, v)) for u, v, keep in H.edges(data=attr) if keep}


def test_global_threshold_filter(weighted_triangle):
    H = global_threshold_filter(weighted_triangle, threshold=2.0)
    assert H[0][1]["global_threshold_keep"] is True
    assert H[0][2]["global_threshold_keep"] is True
    assert H[1][2]["global_threshold_keep"] is False
    H = global_threshold_filter(weighted_triangle, threshold=10.0)
    backbone = boolean_filter(H, "global_threshold_keep")
    assert set(backbone) == set(weighted_triangle)
    assert backbone.number_of_edges() == 0


@pytest.mark.parametrize(
    ("func", "kwargs"),
    [
        (strongest_n_ties, {"n": 0}),
        (global_sparsification, {"s": 0.0}),
        (edge_betweenness_filter, {"s": 1.5}),
        (node_degree_filter, {"min_degree": -1}),
    ],
)
def test_invalid_parameter_raises(func, kwargs, weighted_triangle):
    with pytest.raises(ValueError):
        func(weighted_triangle, **kwargs)


def test_custom_weight_name():
    G = nx.Graph()
    G.add_edge(0, 1, strength=2.0)
    G.add_edge(1, 2, strength=1.0)
    H = metric_backbone(G, weight="strength")
    assert all(keep for *_, keep in H.edges(data="metric_keep"))


class TestStrongestNTiesAndPrimaryLinkage:
    @pytest.fixture
    def digraph(self):
        D = nx.DiGraph()
        D.add_weighted_edges_from(
            [("a", "b", 3), ("a", "c", 1), ("b", "c", 2), ("c", "a", 5), ("b", "a", 1)]
        )
        return D

    def test_strongest_n_ties_undirected(self, weighted_triangle):
        H = strongest_n_ties(weighted_triangle, n=1)
        assert _kept(H, "strongest_n_ties_keep") == {
            frozenset((0, 1)),
            frozenset((0, 2)),
        }

    def test_strongest_n_ties_directed(self, digraph):
        H = strongest_n_ties(digraph, n=1)
        assert _kept(H, "strongest_n_ties_keep") == {
            frozenset(e) for e in [("a", "b"), ("b", "c"), ("c", "a")]
        }
        assert H["b"]["a"]["strongest_n_ties_keep"] is False
        assert H["a"]["b"]["strongest_n_ties_keep"] is True

    def test_strongest_n_ties_directed_n2(self, digraph):
        H = strongest_n_ties(digraph, n=2)
        kept = {(u, v) for u, v, k in H.edges(data="strongest_n_ties_keep") if k}
        assert kept == {("a", "b"), ("a", "c"), ("b", "c"), ("b", "a"), ("c", "a")}

    def test_primary_linkage_directed(self, digraph):
        H = primary_linkage_analysis(digraph)
        kept = {(u, v) for u, v, k in H.edges(data="primary_linkage_keep") if k}
        assert kept == {("a", "b"), ("b", "c"), ("c", "a")}

    def test_primary_linkage_directed_sink(self, digraph):
        digraph.add_edge("c", "sink", weight=0.5)
        digraph.add_edge("sink2", "sink", weight=0.1)
        H = primary_linkage_analysis(digraph)
        kept = {(u, v) for u, v, k in H.edges(data="primary_linkage_keep") if k}
        assert kept == {("a", "b"), ("b", "c"), ("c", "a"), ("sink2", "sink")}

    def test_primary_linkage_undirected_known(self, weighted_triangle):
        H = primary_linkage_analysis(weighted_triangle)
        assert _kept(H, "primary_linkage_keep") == {
            frozenset((0, 1)),
            frozenset((0, 2)),
        }


class TestGlobalAndBetweenness:
    def test_global_sparsification_known(self, weighted_triangle):
        H = global_sparsification(weighted_triangle, s=0.5)  # ceil(0.5 * 3) = 2
        assert _kept(H, "global_sparsification_keep") == {
            frozenset((0, 1)),
            frozenset((0, 2)),
        }
        G = nx.Graph()
        G.add_weighted_edges_from([(0, 1, 5.0), (1, 2, 1.0), (2, 3, 4.0), (3, 0, 2.0)])
        H = global_sparsification(G, s=0.75)
        assert _kept(H, "global_sparsification_keep") == {
            frozenset((0, 1)),
            frozenset((2, 3)),
            frozenset((0, 3)),
        }

    def test_edge_betweenness_uses_inverse_weight_distance(self):
        # Square 0-1-2-3-0; the strong edges make 0-1-2 the shortest route.
        G = nx.Graph()
        G.add_weighted_edges_from(
            [(0, 1, 10.0), (1, 2, 10.0), (2, 3, 1.0), (3, 0, 1.0)]
        )
        H = edge_betweenness_filter(G, s=0.5)
        dist = nx.Graph()
        for u, v, w in G.edges(data="weight"):
            dist.add_edge(u, v, d=1.0 / w)
        expected = nx.edge_betweenness_centrality(dist, weight="d")
        for u, v, eb in H.edges(data="edge_betweenness"):
            assert eb == pytest.approx(expected.get((u, v), expected.get((v, u))))
        assert _kept(H, "edge_betweenness_keep") == {
            frozenset((0, 1)),
            frozenset((1, 2)),
        }

    def test_edge_betweenness_rounding_unweighted_and_nonpositive(
        self, weighted_triangle
    ):
        H = edge_betweenness_filter(weighted_triangle, s=0.5)  # ceil(1.5) = 2
        assert len(_kept(H, "edge_betweenness_keep")) == 2
        H = edge_betweenness_filter(nx.barbell_graph(3, 0), s=0.1)
        assert _kept(H, "edge_betweenness_keep") == {frozenset((2, 3))}
        G = nx.path_graph(3)
        G[0][1]["weight"] = 0.0
        G[1][2]["weight"] = 1.0
        H = edge_betweenness_filter(G, s=1.0)
        assert all(k for *_, k in H.edges(data="edge_betweenness_keep"))


def test_node_degree_filter_known():
    G = nx.path_graph(4)
    H = node_degree_filter(G, min_degree=2)
    assert H.nodes[1]["node_degree_keep"] is True
    assert H.nodes[0]["node_degree_keep"] is False
    backbone = boolean_filter(H, "node_degree_keep")
    assert backbone.number_of_edges() == 1


class TestHighSalienceSkeleton:
    def test_hand_computed_triangle(self, weighted_triangle):
        # distances: d01 = 1/3, d12 = 1, d02 = 1/2.  Every root's tree is
        # {(0, 1), (0, 2)}, because 1 -> 2 is shorter via node 0.
        H = high_salience_skeleton(weighted_triangle)
        assert H[0][1]["salience"] == 1.0
        assert H[0][2]["salience"] == 1.0
        assert H[1][2]["salience"] == 0.0

    def test_c4_ties_split_credit(self):
        G = nx.cycle_graph(4)
        nx.set_edge_attributes(G, 1.0, "weight")
        H = high_salience_skeleton(G)
        saliences = nx.get_edge_attributes(H, "salience")
        assert all(v == pytest.approx(0.75) for v in saliences.values())
        assert sum(saliences.values()) == pytest.approx(G.number_of_nodes() - 1)

    @pytest.mark.parametrize("seed", range(3))
    def test_saliences_sum_to_n_minus_one(self, seed):
        rng = random.Random(seed)
        G = nx.connected_watts_strogatz_graph(25, 4, 0.3, seed=seed)
        for u, v in G.edges():
            G[u][v]["weight"] = rng.choice([1.0, 2.0, 4.0])  # creates ties
        H = high_salience_skeleton(G)
        total = sum(nx.get_edge_attributes(H, "salience").values())
        assert total == pytest.approx(G.number_of_nodes() - 1)


class TestMetricUltrametric:
    def test_semi_metric_triangle(self):
        G = nx.Graph()
        # distances 1, 1 and 2.5: the direct 0-2 edge is semi-metric
        G.add_weighted_edges_from([(0, 1, 1.0), (1, 2, 1.0), (0, 2, 0.4)])
        assert _kept(metric_backbone(G), "metric_keep") == {
            frozenset((0, 1)),
            frozenset((1, 2)),
        }
        assert _kept(ultrametric_backbone(G), "ultrametric_keep") == {
            frozenset((0, 1)),
            frozenset((1, 2)),
        }

    def test_metric_but_not_ultrametric(self):
        G = nx.Graph()
        # direct distance 1/0.6 = 1.67 < 2 (metric) but > max(1, 1) (not ultrametric)
        G.add_weighted_edges_from([(0, 1, 1.0), (1, 2, 1.0), (0, 2, 0.6)])
        assert metric_backbone(G)[0][2]["metric_keep"] is True
        assert ultrametric_backbone(G)[0][2]["ultrametric_keep"] is False

    def test_exact_ties_kept(self):
        G = nx.Graph()
        G.add_weighted_edges_from([(0, 1, 1.0), (1, 2, 1.0), (0, 2, 0.5)])
        assert metric_backbone(G)[0][2]["metric_keep"] is True
        G[0][2]["weight"] = 1.0
        assert ultrametric_backbone(G)[0][2]["ultrametric_keep"] is True

    def test_metric_relative_tolerance(self):
        """Regression: an absolute 1e-12 tolerance missed ties at large scales."""
        rng = random.Random(0)
        for _ in range(2000):
            a, b = rng.randint(1, 999), rng.randint(1, 999)
            scale = 10.0 ** rng.randint(-6, 6)
            G = nx.Graph()
            G.add_edge(0, 1, weight=1 / (a * scale))
            G.add_edge(1, 2, weight=1 / (b * scale))
            G.add_edge(0, 2, weight=1 / ((a + b) * scale))
            assert metric_backbone(G)[0][2]["metric_keep"] is True

    def test_self_loops(self):
        G = nx.Graph()
        G.add_weighted_edges_from([(0, 1, 1.0), (1, 2, 2.0), (0, 0, 3.0)])
        for func, attr in [
            (metric_backbone, "metric_keep"),
            (ultrametric_backbone, "ultrametric_keep"),
        ]:
            H = func(G)
            assert H[0][0][attr] is False
            assert H[0][1][attr] is True and H[1][2][attr] is True


class TestDoublyStochastic:
    def test_row_and_column_sums(self):
        G = nx.complete_graph(5)
        rng = random.Random(1)
        for u, v in G.edges():
            G[u][v]["weight"] = rng.uniform(0.5, 5.0)
        G.add_node("isolated")
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            H = doubly_stochastic_filter(G)
        for node in range(5):
            total = sum(d["ds_weight"] for *_, d in H.edges(node, data=True))
            assert total == pytest.approx(1.0, abs=1e-7)

    def test_single_edge_mixed_labels(self):
        G = nx.Graph()
        G.add_edge("a", 1, weight=2.0)
        H = doubly_stochastic_filter(G)
        assert H["a"][1]["ds_weight"] == pytest.approx(1.0)

    @pytest.mark.parametrize("G", [nx.path_graph(3), nx.star_graph(4)])
    def test_warns_when_not_scalable(self, G):
        nx.set_edge_attributes(G, 1.0, "weight")
        for _ in range(2):  # first call compiles the decorator; check both
            with pytest.warns(RuntimeWarning, match="did not converge") as rec:
                doubly_stochastic_filter(G)
            assert rec[0].filename == __file__


class TestHBackbone:
    def test_all_weights_below_one(self):
        """Regression: h = 0 used to keep every edge (weight >= 0)."""
        G = nx.barbell_graph(4, 0)
        nx.set_edge_attributes(G, 0.5, "weight")
        H = h_backbone(G)
        kept = _kept(H, "h_backbone_keep")
        assert len(kept) < G.number_of_edges()
        # Only the h-bridge part remains: the bridge and the edges into it.
        assert kept == {frozenset((3, 4))} | {frozenset((i, 3)) for i in range(3)} | {
            frozenset((4, j)) for j in range(5, 8)
        }

    def test_bridge_values_and_strength_part(self):
        G = nx.barbell_graph(4, 0)
        nx.set_edge_attributes(G, 0.5, "weight")
        G[0][1]["weight"] = 10.0
        H = h_backbone(G)
        # bridge = ordered-pair edge betweenness / N
        assert H[3][4]["h_bridge"] == pytest.approx(2 * 16 / 8)
        assert H[0][3]["h_bridge"] == pytest.approx(2 * 5 / 8)
        assert H[0][1]["h_bridge"] == pytest.approx(2 * 1 / 8)
        assert H[0][1]["h_backbone_keep"] is True  # h-strength = 1
        assert H[1][2]["h_backbone_keep"] is False
        assert len(_kept(H, "h_backbone_keep")) == 8

    def test_star_known(self):
        star = nx.Graph()
        star.add_weighted_edges_from([(0, 1, 5.0), (0, 2, 3.0), (0, 3, 1.0)])
        # h_s = 2 -> edges with weight >= 2; bridges are 6/4 = 1.5 -> h_b = 1
        H = h_backbone(star)
        assert all(k for *_, k in H.edges(data="h_backbone_keep"))
        assert H[0][1]["h_bridge"] == pytest.approx(1.5)


class TestModularityBackbone:
    # The pinned Les Misérables count (seed=42) is in test_les_miserables.py.

    def test_seed_reproducible(self, karate):
        v1 = nx.get_node_attributes(modularity_backbone(karate, seed=3), "vitality")
        v2 = nx.get_node_attributes(modularity_backbone(karate, seed=3), "vitality")
        assert v1 == v2
        v3 = nx.get_node_attributes(
            modularity_backbone(karate, seed=random.Random(3)), "vitality"
        )
        assert set(v3) == set(v1)

    def test_zero_total_weight(self):
        G = nx.path_graph(3)
        nx.set_edge_attributes(G, 0.0, "weight")
        H = modularity_backbone(G, seed=1)
        assert set(nx.get_node_attributes(H, "vitality").values()) == {0.0}


class TestPMFG:
    @pytest.mark.parametrize("seed", [None, 2])
    def test_maximal_planar_on_complete_graph(self, seed):
        # Equal weights (all ties) and random weights of either sign.
        G = nx.complete_graph(10)
        rng = random.Random(seed)
        for u, v in G.edges():
            G[u][v]["weight"] = 1.0 if seed is None else rng.uniform(-1, 1)
        backbone = boolean_filter(planar_maximally_filtered_graph(G), "pmfg_keep")
        assert backbone.number_of_edges() == 3 * (10 - 2)
        assert nx.check_planarity(backbone)[0]

    def test_small_graph(self, weighted_triangle):
        G = nx.Graph()
        G.add_edge(0, 1, weight=1.0)
        assert planar_maximally_filtered_graph(G)[0][1]["pmfg_keep"] is True
        H = planar_maximally_filtered_graph(weighted_triangle)
        assert all(k for *_, k in H.edges(data="pmfg_keep"))


def test_maximum_spanning_tree_selects_heaviest_edges():
    G = nx.Graph()
    G.add_weighted_edges_from([(0, 1, 5.0), (1, 2, 3.0), (0, 2, 1.0)])
    backbone = boolean_filter(maximum_spanning_tree_backbone(G), "mst_keep")
    assert {frozenset(e) for e in backbone.edges()} == {
        frozenset((0, 1)),
        frozenset((1, 2)),
    }
