"""Known-answer and regression tests for structural backbone methods."""

import random
import warnings

import networkx as nx
import pytest

import networkx_backbone as nb
from networkx_backbone import (
    boolean_filter,
    doubly_stochastic_filter,
    edge_betweenness_filter,
    global_sparsification,
    h_backbone,
    high_salience_skeleton,
    metric_backbone,
    modularity_backbone,
    planar_maximally_filtered_graph,
    primary_linkage_analysis,
    strongest_n_ties,
    ultrametric_backbone,
)

STRUCTURAL = [
    ("global_threshold_filter", {"threshold": 1.0}),
    ("strongest_n_ties", {}),
    ("global_sparsification", {}),
    ("primary_linkage_analysis", {}),
    ("edge_betweenness_filter", {}),
    ("node_degree_filter", {}),
    ("high_salience_skeleton", {}),
    ("metric_backbone", {}),
    ("ultrametric_backbone", {}),
    ("doubly_stochastic_filter", {}),
    ("h_backbone", {}),
    ("modularity_backbone", {"seed": 1}),
    ("planar_maximally_filtered_graph", {}),
    ("maximum_spanning_tree_backbone", {}),
]


def _kept(H, attr):
    return {frozenset((u, v)) for u, v, keep in H.edges(data=attr) if keep}


@pytest.mark.parametrize(("name", "kwargs"), STRUCTURAL, ids=[s[0] for s in STRUCTURAL])
def test_multigraph_not_implemented(name, kwargs):
    M = nx.MultiGraph()
    M.add_edge(0, 1, weight=5.0)
    M.add_edge(0, 1, weight=1.0)
    M.add_edge(1, 2, weight=2.0)
    with pytest.raises(nx.NetworkXNotImplemented):
        getattr(nb, name)(M, **kwargs)


@pytest.mark.parametrize(("name", "kwargs"), STRUCTURAL, ids=[s[0] for s in STRUCTURAL])
def test_empty_graph(name, kwargs):
    for G in (nx.Graph(), nx.empty_graph(3)):
        H = getattr(nb, name)(G, **kwargs)
        assert set(H) == set(G)
        assert H.number_of_edges() == 0


@pytest.mark.parametrize(("name", "kwargs"), STRUCTURAL, ids=[s[0] for s in STRUCTURAL])
def test_does_not_mutate_input(name, kwargs):
    G = nx.karate_club_graph()
    before = {e: dict(d) for e, d in G.edges.items()}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        H = getattr(nb, name)(G, **kwargs)
    assert H is not G
    assert {e: dict(d) for e, d in G.edges.items()} == before


@pytest.mark.parametrize(
    "name",
    [
        "high_salience_skeleton",
        "metric_backbone",
        "ultrametric_backbone",
        "h_backbone",
        "planar_maximally_filtered_graph",
        "doubly_stochastic_filter",
    ],
)
def test_missing_weight_raises_networkx_error(name):
    G = nx.Graph()
    G.add_edge(0, 1, weight=1.0)
    G.add_edge(1, 2)
    with pytest.raises(nx.NetworkXError, match="no 'weight' attribute"):
        getattr(nb, name)(G)


@pytest.mark.parametrize(
    "name", ["high_salience_skeleton", "metric_backbone", "ultrametric_backbone"]
)
@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_non_positive_weight_raises(name, bad):
    G = nx.Graph()
    G.add_edge(0, 1, weight=bad)
    G.add_edge(1, 2, weight=1.0)
    with pytest.raises(nx.NetworkXError, match="strictly positive"):
        getattr(nb, name)(G)


@pytest.mark.parametrize("bad", ["heavy", None, float("nan")])
def test_non_numeric_weight_raises(bad):
    G = nx.Graph()
    G.add_edge(0, 1, weight=bad)
    with pytest.raises(nx.NetworkXError):
        metric_backbone(G)


def test_negative_weight_doubly_stochastic_raises():
    G = nx.Graph()
    G.add_edge(0, 1, weight=-1.0)
    with pytest.raises(nx.NetworkXError, match="negative"):
        doubly_stochastic_filter(G)


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

    def test_primary_linkage_undirected_known(self, weighted_triangle):
        H = primary_linkage_analysis(weighted_triangle)
        assert _kept(H, "primary_linkage_keep") == {frozenset((0, 1)), frozenset((0, 2))}

    def test_mixed_type_nodes_with_tied_weights(self):
        G = nx.Graph()
        G.add_edge("a", 1, weight=1.0)
        G.add_edge("a", "b", weight=1.0)
        G.add_edge(1, (2, 3), weight=1.0)
        H = strongest_n_ties(G, n=1)
        assert len(nx.get_edge_attributes(H, "strongest_n_ties_keep")) == 3
        H = primary_linkage_analysis(G)
        assert len(nx.get_edge_attributes(H, "primary_linkage_keep")) == 3


class TestGlobalAndBetweenness:
    def test_global_sparsification_known(self):
        G = nx.Graph()
        G.add_weighted_edges_from([(0, 1, 5.0), (1, 2, 1.0), (2, 3, 4.0), (3, 0, 2.0)])
        H = global_sparsification(G, s=0.75)
        assert _kept(H, "global_sparsification_keep") == {
            frozenset((0, 1)), frozenset((2, 3)), frozenset((0, 3))
        }

    def test_edge_betweenness_uses_inverse_weight_distance(self):
        # Square 0-1-2-3-0; the strong edges make 0-1-2 the shortest route.
        G = nx.Graph()
        G.add_weighted_edges_from([(0, 1, 10.0), (1, 2, 10.0), (2, 3, 1.0), (3, 0, 1.0)])
        H = edge_betweenness_filter(G, s=0.5)
        dist = nx.Graph()
        for u, v, w in G.edges(data="weight"):
            dist.add_edge(u, v, d=1.0 / w)
        expected = nx.edge_betweenness_centrality(dist, weight="d")
        for u, v, eb in H.edges(data="edge_betweenness"):
            assert eb == pytest.approx(expected.get((u, v), expected.get((v, u))))
        assert _kept(H, "edge_betweenness_keep") == {frozenset((0, 1)), frozenset((1, 2))}

    def test_edge_betweenness_unweighted_and_nonpositive(self):
        H = edge_betweenness_filter(nx.barbell_graph(3, 0), s=0.1)
        assert _kept(H, "edge_betweenness_keep") == {frozenset((2, 3))}
        G = nx.path_graph(3)
        G[0][1]["weight"] = 0.0
        G[1][2]["weight"] = 1.0
        H = edge_betweenness_filter(G, s=1.0)
        assert all(k for *_, k in H.edges(data="edge_betweenness_keep"))


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

    def test_mixed_type_nodes(self):
        G = nx.Graph()
        G.add_edge("a", 1, weight=1.0)
        G.add_edge(1, 2.5, weight=2.0)
        H = high_salience_skeleton(G)
        assert H["a"][1]["salience"] == 1.0


class TestMetricUltrametric:
    def test_semi_metric_triangle(self):
        G = nx.Graph()
        # distances 1, 1 and 2.5: the direct 0-2 edge is semi-metric
        G.add_weighted_edges_from([(0, 1, 1.0), (1, 2, 1.0), (0, 2, 0.4)])
        assert _kept(metric_backbone(G), "metric_keep") == {
            frozenset((0, 1)), frozenset((1, 2))
        }
        assert _kept(ultrametric_backbone(G), "ultrametric_keep") == {
            frozenset((0, 1)), frozenset((1, 2))
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
        for func, attr in [(metric_backbone, "metric_keep"),
                           (ultrametric_backbone, "ultrametric_keep")]:
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

    @pytest.mark.parametrize("G", [nx.path_graph(3), nx.star_graph(4)])
    def test_warns_when_not_scalable(self, G):
        nx.set_edge_attributes(G, 1.0, "weight")
        with pytest.warns(RuntimeWarning, match="did not converge"):
            doubly_stochastic_filter(G)

    def test_mixed_type_nodes(self):
        G = nx.Graph()
        G.add_edge("a", 1, weight=2.0)
        H = doubly_stochastic_filter(G)
        assert H["a"][1]["ds_weight"] == pytest.approx(1.0)


class TestHBackbone:
    def test_all_weights_below_one(self):
        """Regression: h = 0 used to keep every edge (weight >= 0)."""
        G = nx.barbell_graph(4, 0)
        nx.set_edge_attributes(G, 0.5, "weight")
        H = h_backbone(G)
        kept = _kept(H, "h_backbone_keep")
        assert len(kept) < G.number_of_edges()
        # Only the h-bridge part remains: the bridge and the edges into it.
        assert kept == {frozenset((3, 4))} | {
            frozenset((i, 3)) for i in range(3)
        } | {frozenset((4, j)) for j in range(5, 8)}

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

    def test_star_known(self, weighted_star):
        # h_s = 2 -> edges with weight >= 2; bridges are 6/4 = 1.5 -> h_b = 1
        H = h_backbone(weighted_star)
        assert all(k for *_, k in H.edges(data="h_backbone_keep"))
        assert H[0][1]["h_bridge"] == pytest.approx(1.5)


class TestModularityBackbone:
    def test_seed_reproducible(self, karate):
        v1 = nx.get_node_attributes(modularity_backbone(karate, seed=3), "vitality")
        v2 = nx.get_node_attributes(modularity_backbone(karate, seed=3), "vitality")
        assert v1 == v2
        v3 = nx.get_node_attributes(
            modularity_backbone(karate, seed=random.Random(3)), "vitality"
        )
        assert set(v3) == set(v1)

    def test_les_miserables_seed_42_matches_previous_default(self, les_miserables_weighted):
        H = modularity_backbone(les_miserables_weighted, seed=42)
        assert boolean_filter(H, "modularity_keep").number_of_edges() == 72

    def test_zero_total_weight(self):
        G = nx.path_graph(3)
        nx.set_edge_attributes(G, 0.0, "weight")
        H = modularity_backbone(G, seed=1)
        assert set(nx.get_node_attributes(H, "vitality").values()) == {0.0}


class TestPMFG:
    def test_negative_weights_allowed(self):
        G = nx.complete_graph(6)
        rng = random.Random(2)
        for u, v in G.edges():
            G[u][v]["weight"] = rng.uniform(-1, 1)
        H = planar_maximally_filtered_graph(G)
        backbone = boolean_filter(H, "pmfg_keep")
        assert backbone.number_of_edges() == 3 * (6 - 2)
        assert nx.check_planarity(backbone)[0]

    def test_small_graph(self):
        G = nx.Graph()
        G.add_edge(0, 1, weight=1.0)
        assert planar_maximally_filtered_graph(G)[0][1]["pmfg_keep"] is True


def test_directed_not_implemented():
    D = nx.DiGraph()
    D.add_edge(0, 1, weight=1.0)
    for name in ["metric_backbone", "h_backbone", "doubly_stochastic_filter"]:
        with pytest.raises(nx.NetworkXNotImplemented):
            getattr(nb, name)(D)
