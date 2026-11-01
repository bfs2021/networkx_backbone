"""Regression tests for filters and measures found in code review."""

import math

import networkx as nx
import pytest

import networkx_backbone as nb


class TestAdjustPvaluesMissing:
    @pytest.mark.parametrize("method", ["bonferroni", "holm", "hochberg", "bh", "by"])
    def test_nan_is_kept_and_not_counted(self, method):
        with_nan = nb.adjust_pvalues([0.01, math.nan, 0.04, 0.03], method)
        without = nb.adjust_pvalues([0.01, 0.04, 0.03], method)
        assert math.isnan(with_nan[1])
        assert [with_nan[0], with_nan[2], with_nan[3]] == pytest.approx(without)

    def test_bonferroni_clips_to_unit_interval(self):
        assert nb.adjust_pvalues([-0.2, 0.4], "bonferroni") == [0.0, 0.8]

    def test_hochberg_matches_r(self):
        # R: p.adjust(c(0.01, 0.02, 0.03, 0.5), "hochberg")
        assert nb.adjust_pvalues([0.01, 0.02, 0.03, 0.5], "hochberg") == pytest.approx(
            [0.04, 0.06, 0.06, 0.5]
        )

    def test_unknown_method(self):
        with pytest.raises(ValueError, match="method must be one of"):
            nb.adjust_pvalues([0.1], "hommel")

    def test_empty(self):
        assert nb.adjust_pvalues([], "bh") == []


@pytest.fixture
def scored():
    G = nx.Graph(name="toy")
    G.add_node("a", color="red")
    G.add_node("b", color="blue")
    G.add_node("c", color="green")
    G.add_edge("a", "b", p=0.01, keep=True)
    G.add_edge("b", "c", p=0.5, keep=False)
    return G


class TestAttributePreservation:
    def test_threshold_filter_keeps_graph_and_node_attrs(self, scored):
        H = nb.threshold_filter(scored, "p", 0.05, include_all_nodes=False)
        assert H.graph["name"] == "toy"
        assert dict(H.nodes(data=True)) == {"a": {"color": "red"}, "b": {"color": "blue"}}

    def test_boolean_filter_keeps_graph_attrs(self, scored):
        assert nb.boolean_filter(scored, "keep").graph["name"] == "toy"

    def test_fraction_filter_keeps_graph_attrs(self, scored):
        assert nb.fraction_filter(scored, "p", 0.5).graph["name"] == "toy"

    def test_threshold_drops_nan_scores(self, scored):
        scored["b"]["c"]["p"] = math.nan
        H = nb.threshold_filter(scored, "p", 1.0)
        assert list(H.edges()) == [("a", "b")]


class TestFractionFilterUnscored:
    def test_unscored_edges_never_kept(self):
        G = nx.Graph()
        G.add_edge(0, 1, s=0.1)
        G.add_edge(1, 2)
        assert list(nb.fraction_filter(G, "s", 1.0).edges()) == [(0, 1)]

    def test_nodes_mode(self):
        G = nx.path_graph(4)
        nx.set_node_attributes(G, {0: 3.0, 1: 1.0, 2: 2.0, 3: 0.5}, "s")
        assert set(nb.fraction_filter(G, "s", 0.5, filter_on="nodes")) == {3, 1}
        assert set(nb.fraction_filter(G, "s", 0.5, ascending=False, filter_on="nodes")) == {0, 2}

    def test_invalid_filter_on(self):
        with pytest.raises(ValueError):
            nb.fraction_filter(nx.path_graph(2), "s", 0.5, filter_on="faces")

    def test_tie_break_is_input_order(self):
        G = nx.Graph()
        G.add_edges_from([(0, 1), (2, 3), (4, 5)], s=1.0)
        assert list(nb.fraction_filter(G, "s", 0.5).edges()) == [(0, 1)]


class TestMultigraphFilters:
    def test_threshold_preserves_keys(self):
        M = nx.MultiGraph()
        M.add_edge(0, 1, key="x", p=0.01)
        M.add_edge(0, 1, key="y", p=0.9)
        H = nb.threshold_filter(M, "p", 0.05)
        assert list(H.edges(keys=True)) == [(0, 1, "x")]

    @pytest.mark.parametrize(
        ("aggregate", "expected"),
        [("count", 2), ("sum", 30), ("max", 20), ("min", 10), ("mean", 15)],
    )
    def test_multigraph_to_weighted_aggregate(self, aggregate, expected):
        M = nx.MultiGraph([(0, 1, {"weight": 10}), (0, 1, {"weight": 20})])
        assert nb.multigraph_to_weighted(M, aggregate=aggregate)[0][1]["weight"] == expected

    def test_multigraph_to_weighted_bad_aggregate(self):
        with pytest.raises(ValueError):
            nb.multigraph_to_weighted(nx.MultiGraph(), aggregate="median")


class TestConsensusBackbone:
    def test_mixed_type_nodes(self):
        B1 = nx.Graph([(1, "a"), (2, "b")])
        B2 = nx.Graph([("a", 1)])
        assert list(nb.consensus_backbone(B1, B2).edges()) == [(1, "a")]

    def test_order_follows_first_backbone(self):
        edges = [(i, i + 1) for i in range(20)]
        B1 = nx.Graph(edges)
        B2 = nx.Graph(list(reversed(edges)))
        assert list(nb.consensus_backbone(B1, B2).edges()) == edges

    def test_directed_respects_direction(self):
        B1 = nx.DiGraph([(0, 1), (1, 2)])
        B2 = nx.DiGraph([(1, 0), (1, 2)])
        assert list(nb.consensus_backbone(B1, B2).edges()) == [(1, 2)]

    def test_multigraph(self):
        M1 = nx.MultiGraph([(0, 1), (1, 2)])
        M2 = nx.MultiGraph([(0, 1)])
        assert list(nb.consensus_backbone(M1, M2).edges()) == [(0, 1)]


class TestMeasures:
    def test_reachability_directed_path(self):
        assert nb.reachability(nx.DiGraph([(0, 1), (1, 2)])) == pytest.approx(0.5)

    def test_reachability_directed_cycle(self):
        assert nb.reachability(nx.cycle_graph(4, create_using=nx.DiGraph)) == 1.0

    def test_reachability_directed_dag_between_cycles(self):
        # SCC {0,1} -> SCC {2,3}: 2*1 + 2*1 within, 2*2 across = 8 of 12.
        G = nx.DiGraph([(0, 1), (1, 0), (2, 3), (3, 2), (1, 2)])
        assert nb.reachability(G) == pytest.approx(8 / 12)

    def test_reachability_trivial(self):
        assert nb.reachability(nx.empty_graph(1)) == 1.0

    def test_compare_backbones_forwards_weight(self):
        pytest.importorskip("scipy")
        G = nx.Graph()
        G.add_edge(0, 1, w=1)
        G.add_edge(1, 2, w=5)
        H = nx.Graph()
        H.add_edge(1, 2, w=5)
        out = nb.compare_backbones(G, {"h": H}, measures=[nb.ks_weight], weight="w")
        assert out["h"]["ks_weight"] == pytest.approx(nb.ks_weight(G, H, weight="w"))

    def test_compare_backbones_default_measures(self, les_miserables_weighted):
        bb = nb.threshold_filter(
            nb.disparity_filter(les_miserables_weighted), "disparity_pvalue", 0.05
        )
        out = nb.compare_backbones(les_miserables_weighted, {"d": bb})
        assert set(out["d"]) == {"node_fraction", "edge_fraction", "weight_fraction"}

    def test_compare_backbones_accepts_lambdas(self, les_miserables_weighted):
        out = nb.compare_backbones(
            les_miserables_weighted,
            {"same": les_miserables_weighted},
            measures=[lambda G, H: 1.0],
        )
        assert list(out["same"].values()) == [1.0]

    def test_empty_graph_measures(self):
        E = nx.Graph()
        assert nb.edge_fraction(E, E) == 0.0
        assert nb.node_fraction(E, E) == 0.0
        assert nb.weight_fraction(E, E) == 0.0
