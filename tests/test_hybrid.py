"""Known-answer tests for networkx_backbone.hybrid (GLAB).

Shared behaviour (attributes, weight validation, directed/multigraph
rejection, empty graphs, no input mutation) is covered by
``test_api_contract.py``.
"""

import networkx as nx
import pytest

from networkx_backbone import glab_filter


class TestGLABPaperEquations:
    """Known answers from Zhang et al. (2014), Eqs. 1 and 4."""

    def test_involvement_sums_to_one_per_node(self, les_miserables_weighted):
        from networkx_backbone.hybrid import _involvement

        G = les_miserables_weighted

        def length(u, v, d):
            return 1.0 / d["weight"]

        for node in list(G)[:10]:
            inv = _involvement(G, node, length)
            assert sum(inv.values()) == pytest.approx(1.0)

    def test_involvement_splits_tied_paths(self):
        from networkx_backbone.hybrid import _involvement

        # From node 0 in C4: node 1 via (0,1), node 3 via (0,3), node 2 split.
        assert _involvement(nx.cycle_graph(4), 0, None) == {1: 0.5, 3: 0.5}

    def test_cycle_hand_computed(self):
        # Every node has k=2 and I=1/2, so SI = (1 - 1/2) ** (1 ** c) = 0.5.
        H = glab_filter(nx.cycle_graph(4), weight=None, c=0.7)
        assert all(
            d["glab_pvalue"] == pytest.approx(0.5) for *_, d in H.edges(data=True)
        )

    def test_star_leaf_uses_hub_side(self):
        # Hub of degree 3: I = 1/3 for each spoke, SI = (2/3) ** (2 ** c).
        H = glab_filter(nx.star_graph(3), weight=None, c=1.0)
        for *_, d in H.edges(data=True):
            assert d["glab_pvalue"] == pytest.approx((2 / 3) ** 2)

    def test_c_zero_ignores_degree(self):
        H = glab_filter(nx.star_graph(5), weight=None, c=0.0)
        for *_, d in H.edges(data=True):
            assert d["glab_pvalue"] == pytest.approx(0.8)

    def test_isolated_dyad_is_not_significant(self):
        G = nx.Graph()
        G.add_edge("a", "b", weight=3.0)
        H = glab_filter(G)
        assert H["a"]["b"]["glab_pvalue"] == 1.0

    def test_heavy_link_more_important(self):
        # Path 0-1 is heavy, so 2 reaches 0 through 1 and the heavy link
        # dominates node 1's involvement.
        G = nx.Graph()
        G.add_weighted_edges_from([(0, 1, 10.0), (1, 2, 1.0), (0, 2, 0.1), (2, 3, 1.0)])
        H = glab_filter(G)
        assert H[0][1]["glab_pvalue"] < H[0][2]["glab_pvalue"]

    def test_les_miserables_reference(self, les_miserables_weighted):
        H = glab_filter(les_miserables_weighted)
        assert H["Valjean"]["Cosette"]["glab_pvalue"] == pytest.approx(0.0425, abs=1e-4)
        # The pinned count of edges with p < 0.05 is in test_les_miserables.py.

    def test_negative_c_raises(self, weighted_triangle):
        with pytest.raises(ValueError):
            glab_filter(weighted_triangle, c=-1)

    def test_disconnected_graph(self):
        G = nx.disjoint_union(nx.cycle_graph(4), nx.path_graph(3))
        H = glab_filter(G, weight=None)
        assert all(0.0 <= d["glab_pvalue"] <= 1.0 for *_, d in H.edges(data=True))
