"""Tests for networkx_backbone.measures."""

import networkx as nx
import pytest

from networkx_backbone import (
    compare_backbones,
    disparity_filter,
    edge_fraction,
    ks_degree,
    ks_weight,
    node_fraction,
    reachability,
    threshold_filter,
    weight_fraction,
)


def test_node_fraction():
    G = nx.path_graph(4)
    assert node_fraction(G, G) == 1.0
    assert node_fraction(G, nx.path_graph(3)) == 0.75


def test_edge_fraction():
    G = nx.complete_graph(4)  # 6 edges
    assert edge_fraction(G, G) == 1.0
    assert edge_fraction(G, nx.path_graph(4)) == 0.5  # 3 edges


def test_weight_fraction(weighted_triangle):
    H = nx.Graph()
    H.add_weighted_edges_from([(0, 1, 3.0)])
    assert weight_fraction(weighted_triangle, H) == 0.5


def test_empty_graph_measures():
    E = nx.Graph()
    assert edge_fraction(E, E) == 0.0
    assert node_fraction(E, E) == 0.0
    assert weight_fraction(E, E) == 0.0


def test_ks_identical_graphs_is_zero(weighted_triangle):
    assert ks_degree(nx.complete_graph(5), nx.complete_graph(5)) == 0.0
    assert ks_weight(weighted_triangle, weighted_triangle) == 0.0


class TestReachability:
    def test_undirected(self):
        assert reachability(nx.path_graph(4)) == 1.0
        assert reachability(nx.empty_graph(3)) == 0.0
        assert reachability(nx.empty_graph(1)) == 1.0

    def test_directed_path(self):
        assert reachability(nx.DiGraph([(0, 1), (1, 2)])) == pytest.approx(0.5)

    def test_directed_cycle(self):
        assert reachability(nx.cycle_graph(4, create_using=nx.DiGraph)) == 1.0

    def test_directed_dag_between_cycles(self):
        # SCC {0,1} -> SCC {2,3}: 2*1 + 2*1 within, 2*2 across = 8 of 12.
        G = nx.DiGraph([(0, 1), (1, 0), (2, 3), (3, 2), (1, 2)])
        assert reachability(G) == pytest.approx(8 / 12)


class TestCompareBackbones:
    def test_returns_named_measures(self):
        G = nx.complete_graph(5)
        results = compare_backbones(G, {"path": nx.path_graph(5)}, [edge_fraction])
        assert results == {"path": {"edge_fraction": 0.4}}

    def test_forwards_weight(self):
        pytest.importorskip("scipy")
        G = nx.Graph()
        G.add_edge(0, 1, w=1)
        G.add_edge(1, 2, w=5)
        H = nx.Graph()
        H.add_edge(1, 2, w=5)
        out = compare_backbones(G, {"h": H}, measures=[ks_weight], weight="w")
        assert out["h"]["ks_weight"] == pytest.approx(ks_weight(G, H, weight="w"))

    def test_default_measures(self, les_miserables_weighted):
        bb = threshold_filter(
            disparity_filter(les_miserables_weighted), "disparity_pvalue", 0.05
        )
        out = compare_backbones(les_miserables_weighted, {"d": bb})
        assert set(out["d"]) == {"node_fraction", "edge_fraction", "weight_fraction"}

    def test_accepts_lambdas(self, les_miserables_weighted):
        out = compare_backbones(
            les_miserables_weighted,
            {"same": les_miserables_weighted},
            measures=[lambda G, H: 1.0],
        )
        assert list(out["same"].values()) == [1.0]
