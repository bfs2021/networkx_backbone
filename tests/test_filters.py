"""Tests for networkx_backbone.filters."""

import math

import networkx as nx
import pytest

from networkx_backbone import (
    adjust_pvalues,
    boolean_filter,
    consensus_backbone,
    disparity_filter,
    fraction_filter,
    multigraph_to_weighted,
    threshold_filter,
)


@pytest.fixture
def scored():
    G = nx.Graph(name="toy")
    G.add_node("a", color="red")
    G.add_node("b", color="blue")
    G.add_node("c", color="green")
    G.add_edge("a", "b", p=0.01, keep=True)
    G.add_edge("b", "c", p=0.5, keep=False)
    return G


class TestAdjustPvalues:
    PS = [0.001, 0.008, 0.012, 0.03, 0.04, 0.2, 0.5, 0.9]

    def test_bonferroni(self):
        assert adjust_pvalues([0.01, 0.02, 0.5], "bonferroni") == [0.03, 0.06, 1.0]

    def test_holm_step_down(self):
        assert adjust_pvalues([0.01, 0.02, 0.5], "holm") == pytest.approx(
            [0.03, 0.04, 0.5]
        )

    def test_hochberg_matches_r(self):
        # R: p.adjust(c(0.01, 0.02, 0.03, 0.5), "hochberg")
        assert adjust_pvalues([0.01, 0.02, 0.03, 0.5], "hochberg") == pytest.approx(
            [0.04, 0.06, 0.06, 0.5]
        )

    @pytest.mark.parametrize("method", ["bh", "fdr", "by"])
    def test_matches_scipy(self, method):
        sp = pytest.importorskip("scipy.stats")
        import numpy as np

        expected = sp.false_discovery_control(
            self.PS, method="by" if method == "by" else "bh"
        )
        assert np.allclose(adjust_pvalues(self.PS, method), expected)

    def test_clips_to_unit_interval(self):
        assert adjust_pvalues([0.2, 1.5, -0.1], "none") == [0.2, 1.0, 0.0]
        assert adjust_pvalues([-0.2, 0.4], "bonferroni") == [0.0, 0.8]

    def test_correction_is_conservative(self):
        raw = self.PS
        for method in ("bonferroni", "holm", "hochberg", "bh", "by"):
            adj = adjust_pvalues(raw, method)
            assert len(adj) == len(raw)
            assert all(r - 1e-12 <= a <= 1.0 for a, r in zip(adj, raw)), method

    @pytest.mark.parametrize("method", ["bonferroni", "holm", "hochberg", "bh", "by"])
    def test_nan_is_kept_and_not_counted(self, method):
        with_nan = adjust_pvalues([0.01, math.nan, 0.04, 0.03], method)
        without = adjust_pvalues([0.01, 0.04, 0.03], method)
        assert math.isnan(with_nan[1])
        assert [with_nan[0], with_nan[2], with_nan[3]] == pytest.approx(without)

    def test_empty(self):
        assert adjust_pvalues([], "bh") == []

    def test_unknown_method(self):
        with pytest.raises(ValueError, match="method must be one of"):
            adjust_pvalues([0.1], "hommel")


class TestThresholdFilter:
    def test_below_and_above_partition_edges(self, les_miserables_weighted):
        H = disparity_filter(les_miserables_weighted)
        below = threshold_filter(H, "disparity_pvalue", 0.3, mode="below")
        above = threshold_filter(H, "disparity_pvalue", 0.3, mode="above")
        assert set(below) == set(above) == set(H)
        assert all(p < 0.3 for *_, p in below.edges(data="disparity_pvalue"))
        assert all(p >= 0.3 for *_, p in above.edges(data="disparity_pvalue"))
        assert below.number_of_edges() + above.number_of_edges() == len(H.edges)

    def test_invalid_mode_raises(self, weighted_triangle):
        with pytest.raises(ValueError):
            threshold_filter(weighted_triangle, "weight", 1.0, mode="invalid")

    def test_removes_isolates_on_edge_filter(self, weighted_triangle):
        H = disparity_filter(weighted_triangle)
        filtered = threshold_filter(
            H,
            "disparity_pvalue",
            0.0,  # keep no edges in below mode
            mode="below",
            include_all_nodes=False,
        )
        assert filtered.number_of_edges() == 0
        assert filtered.number_of_nodes() == 0

    def test_node_filter(self):
        G = nx.Graph()
        G.add_node(0, score=0.1)
        G.add_node(1, score=0.9)
        G.add_node(2, score=0.5)
        G.add_edge(0, 1)
        G.add_edge(1, 2)
        filtered = threshold_filter(G, "score", 0.5, mode="below", filter_on="nodes")
        assert 0 in filtered.nodes()
        assert 1 not in filtered.nodes()

    def test_node_filter_remove_isolates_option(self):
        G = nx.Graph()
        G.add_node(0, score=0.1)
        G.add_node(1, score=0.1)
        G.add_node(2, score=0.9)
        G.add_edge(0, 2)

        kept_with_isolates = threshold_filter(
            G, "score", 0.5, mode="below", filter_on="nodes", include_all_nodes=True
        )
        assert set(kept_with_isolates.nodes()) == {0, 1}

        kept_without_isolates = threshold_filter(
            G, "score", 0.5, mode="below", filter_on="nodes", include_all_nodes=False
        )
        assert set(kept_without_isolates.nodes()) == set()

    def test_keeps_graph_and_node_attrs(self, scored):
        H = threshold_filter(scored, "p", 0.05, include_all_nodes=False)
        assert H.graph["name"] == "toy"
        assert dict(H.nodes(data=True)) == {
            "a": {"color": "red"},
            "b": {"color": "blue"},
        }

    def test_drops_nan_scores(self, scored):
        scored["b"]["c"]["p"] = math.nan
        H = threshold_filter(scored, "p", 1.0)
        assert list(H.edges()) == [("a", "b")]

    def test_multigraph_preserves_keys(self):
        M = nx.MultiGraph()
        M.add_edge(0, 1, key="x", p=0.01)
        M.add_edge(0, 1, key="y", p=0.9)
        H = threshold_filter(M, "p", 0.05)
        assert list(H.edges(keys=True)) == [(0, 1, "x")]


class TestThresholdFilterMTC:
    def test_mtc_reduces_or_equal_edges(self, les_miserables_weighted):
        H = disparity_filter(les_miserables_weighted)
        raw = threshold_filter(H, "disparity_pvalue", 0.05).number_of_edges()
        for method in ("bonferroni", "bh"):
            corrected = threshold_filter(
                H, "disparity_pvalue", 0.05, mtc=method
            ).number_of_edges()
            assert corrected <= raw

    def test_mtc_requires_below_mode(self, les_miserables_weighted):
        H = disparity_filter(les_miserables_weighted)
        with pytest.raises(ValueError):
            threshold_filter(H, "disparity_pvalue", 0.05, mode="above", mtc="bh")

    def test_mtc_none_matches_plain(self, les_miserables_weighted):
        H = disparity_filter(les_miserables_weighted)
        a = threshold_filter(H, "disparity_pvalue", 0.3).number_of_edges()
        b = threshold_filter(H, "disparity_pvalue", 0.3, mtc="none").number_of_edges()
        assert a == b


class TestFractionFilter:
    def test_keeps_lowest_fraction(self, les_miserables_weighted):
        H = disparity_filter(les_miserables_weighted)
        filtered = fraction_filter(H, "disparity_pvalue", 0.5)
        assert filtered.number_of_edges() == math.ceil(0.5 * H.number_of_edges())
        cutoff = max(p for *_, p in filtered.edges(data="disparity_pvalue"))
        dropped = set(H.edges()) - set(filtered.edges())
        assert all(H.edges[e]["disparity_pvalue"] >= cutoff for e in dropped)
        all_kept = fraction_filter(H, "disparity_pvalue", 1.0)
        assert all_kept.number_of_edges() == H.number_of_edges()

    def test_invalid_fraction_raises(self, weighted_triangle):
        with pytest.raises(ValueError):
            fraction_filter(weighted_triangle, "weight", 0.0)

    def test_keeps_graph_attrs(self, scored):
        assert fraction_filter(scored, "p", 0.5).graph["name"] == "toy"

    def test_unscored_edges_never_kept(self):
        G = nx.Graph()
        G.add_edge(0, 1, s=0.1)
        G.add_edge(1, 2)
        assert list(fraction_filter(G, "s", 1.0).edges()) == [(0, 1)]

    def test_nodes_mode(self):
        G = nx.path_graph(4)
        nx.set_node_attributes(G, {0: 3.0, 1: 1.0, 2: 2.0, 3: 0.5}, "s")
        assert set(fraction_filter(G, "s", 0.5, filter_on="nodes")) == {3, 1}
        assert set(
            fraction_filter(G, "s", 0.5, ascending=False, filter_on="nodes")
        ) == {0, 2}

    def test_invalid_filter_on(self):
        with pytest.raises(ValueError):
            fraction_filter(nx.path_graph(2), "s", 0.5, filter_on="faces")

    def test_tie_break_is_input_order(self):
        G = nx.Graph()
        G.add_edges_from([(0, 1), (2, 3), (4, 5)], s=1.0)
        assert list(fraction_filter(G, "s", 0.5).edges()) == [(0, 1)]


class TestBooleanFilter:
    def test_filters_by_boolean_and_keeps_nodes(self):
        G = nx.Graph(name="toy")
        G.add_node(5)
        G.add_edge(0, 1, keep=True)
        G.add_edge(1, 2, keep=False)
        G.add_edge(2, 3, keep=True)
        H = boolean_filter(G, "keep")
        assert sorted(H.edges()) == [(0, 1), (2, 3)]
        assert set(H) == set(G)
        assert H.graph["name"] == "toy"


class TestMultigraphToWeighted:
    def test_multigraph_parallel_edge_count(self):
        G = nx.MultiGraph()
        G.add_edge(0, 1)
        G.add_edge(0, 1)
        G.add_edge(0, 1)

        H = multigraph_to_weighted(G)
        assert isinstance(H, nx.Graph)
        assert H.number_of_edges() == 1
        assert H[0][1]["weight"] == 3
        assert H[0][1]["edge_count"] == 3

    def test_multigraph_edge_type_count(self):
        G = nx.MultiGraph()
        G.add_edge(0, 1, edge_type="a")
        G.add_edge(0, 1, edge_type="b")
        G.add_edge(0, 1, edge_type="b")

        H = multigraph_to_weighted(G, edge_type_attr="edge_type")
        assert H[0][1]["weight"] == 2
        assert H[0][1]["edge_count"] == 3
        assert H[0][1]["edge_type_count"] == 2

    def test_multidigraph_direction_preserved(self):
        G = nx.MultiDiGraph()
        G.add_edge(0, 1)
        G.add_edge(0, 1)
        G.add_edge(1, 0)

        H = multigraph_to_weighted(G)
        assert isinstance(H, nx.DiGraph)
        assert H.number_of_edges() == 2
        assert H[0][1]["weight"] == 2
        assert H[1][0]["weight"] == 1

    def test_simple_graph_fills_missing_weight(self):
        G = nx.Graph()
        G.add_edge(0, 1)

        H = multigraph_to_weighted(G)
        assert H[0][1]["weight"] == 1

    @pytest.mark.parametrize(
        ("aggregate", "expected"),
        [("count", 2), ("sum", 30), ("max", 20), ("min", 10), ("mean", 15)],
    )
    def test_aggregate(self, aggregate, expected):
        M = nx.MultiGraph([(0, 1, {"weight": 10}), (0, 1, {"weight": 20})])
        assert (
            multigraph_to_weighted(M, aggregate=aggregate)[0][1]["weight"] == expected
        )

    def test_bad_aggregate(self):
        with pytest.raises(ValueError):
            multigraph_to_weighted(nx.MultiGraph(), aggregate="median")


class TestConsensusBackbone:
    def test_intersection_of_edges(self):
        G1 = nx.Graph([(0, 1), (1, 2)])
        G2 = nx.Graph([(0, 1), (2, 3)])
        assert sorted(consensus_backbone(G1, G2).edges()) == [(0, 1)]
        G3 = nx.Graph([(2, 3)])
        assert consensus_backbone(nx.Graph([(0, 1)]), G3).number_of_edges() == 0

    def test_requires_at_least_two(self):
        with pytest.raises(ValueError):
            consensus_backbone(nx.Graph([(0, 1)]))

    def test_mixed_type_nodes(self):
        B1 = nx.Graph([(1, "a"), (2, "b")])
        B2 = nx.Graph([("a", 1)])
        assert list(consensus_backbone(B1, B2).edges()) == [(1, "a")]

    def test_order_follows_first_backbone(self):
        edges = [(i, i + 1) for i in range(20)]
        B1 = nx.Graph(edges)
        B2 = nx.Graph(list(reversed(edges)))
        assert list(consensus_backbone(B1, B2).edges()) == edges

    def test_directed_respects_direction(self):
        B1 = nx.DiGraph([(0, 1), (1, 2)])
        B2 = nx.DiGraph([(1, 0), (1, 2)])
        assert list(consensus_backbone(B1, B2).edges()) == [(1, 2)]

    def test_multigraph(self):
        M1 = nx.MultiGraph([(0, 1), (1, 2)])
        M2 = nx.MultiGraph([(0, 1)])
        assert list(consensus_backbone(M1, M2).edges()) == [(0, 1)]
