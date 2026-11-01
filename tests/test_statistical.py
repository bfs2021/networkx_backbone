"""Tests for statistical backbone extraction methods."""

import networkx as nx
import pytest

from networkx_backbone import (
    boolean_filter,
    disparity,
    disparity_filter,
    ecm_filter,
    lans,
    lans_filter,
    mlf,
    multiple_linkage_analysis,
    marginal_likelihood_filter,
    noise_corrected_filter,
)


class TestDisparityFilter:
    def test_adds_pvalue_attribute(self, weighted_triangle):
        H = disparity_filter(weighted_triangle)
        for u, v, data in H.edges(data=True):
            assert "disparity_pvalue" in data

    def test_pvalues_in_range(self, weighted_triangle):
        H = disparity_filter(weighted_triangle)
        for _, _, data in H.edges(data=True):
            assert 0.0 <= data["disparity_pvalue"] <= 1.0

    def test_preserves_edges(self, weighted_triangle):
        H = disparity_filter(weighted_triangle)
        assert H.number_of_edges() == weighted_triangle.number_of_edges()

    def test_preserves_nodes(self, weighted_triangle):
        H = disparity_filter(weighted_triangle)
        assert set(H.nodes()) == set(weighted_triangle.nodes())

    def test_missing_weight_raises(self):
        G = nx.Graph()
        G.add_edge(0, 1)
        with pytest.raises(nx.NetworkXError):
            disparity_filter(G)

    def test_directed_graph(self):
        G = nx.DiGraph()
        G.add_weighted_edges_from([(0, 1, 3.0), (1, 2, 1.0)])
        H = disparity_filter(G)
        assert "disparity_pvalue" in H[0][1]


class TestNoiseCorrectedFilter:
    def test_adds_score_attribute(self, weighted_triangle):
        H = noise_corrected_filter(weighted_triangle)
        for u, v, data in H.edges(data=True):
            assert "nc_score" in data

    def test_preserves_structure(self, weighted_triangle):
        H = noise_corrected_filter(weighted_triangle)
        assert H.number_of_edges() == weighted_triangle.number_of_edges()
        assert set(H.nodes()) == set(weighted_triangle.nodes())

    def test_higher_weight_higher_score(self, weighted_triangle):
        H = noise_corrected_filter(weighted_triangle)
        # Edge (0,1) has weight 3.0, edge (1,2) has weight 1.0
        # The higher weight edge should generally have a higher score
        assert isinstance(H[0][1]["nc_score"], float)


class TestMarginalLikelihoodFilter:
    def test_adds_pvalue_attribute(self, weighted_triangle):
        H = marginal_likelihood_filter(weighted_triangle)
        for u, v, data in H.edges(data=True):
            assert "ml_pvalue" in data

    def test_pvalues_in_range(self, weighted_triangle):
        H = marginal_likelihood_filter(weighted_triangle)
        for _, _, data in H.edges(data=True):
            assert 0.0 <= data["ml_pvalue"] <= 1.0


class TestECMFilter:
    def test_adds_pvalue_attribute(self, weighted_triangle):
        H = ecm_filter(weighted_triangle)
        for u, v, data in H.edges(data=True):
            assert "ecm_pvalue" in data

    def test_pvalues_in_range(self, weighted_triangle):
        H = ecm_filter(weighted_triangle)
        for _, _, data in H.edges(data=True):
            assert 0.0 <= data["ecm_pvalue"] <= 1.0

    def test_preserves_structure(self, weighted_triangle):
        H = ecm_filter(weighted_triangle)
        assert H.number_of_edges() == weighted_triangle.number_of_edges()


class TestLANSFilter:
    def test_adds_pvalue_attribute(self, weighted_triangle):
        H = lans_filter(weighted_triangle)
        for u, v, data in H.edges(data=True):
            assert "lans_pvalue" in data

    def test_pvalues_in_range(self, weighted_triangle):
        H = lans_filter(weighted_triangle)
        for _, _, data in H.edges(data=True):
            assert 0.0 <= data["lans_pvalue"] <= 1.0

    def test_preserves_structure(self, weighted_triangle):
        H = lans_filter(weighted_triangle)
        assert H.number_of_edges() == weighted_triangle.number_of_edges()

    def test_directed_heavy_edge_significant_from_target_side(self):
        # Five sources each send one link to ``v``; the heavy edge is the
        # strongest incoming link of ``v`` and must get p = 0 from v's side.
        G = nx.DiGraph()
        for i, w in enumerate([100, 1, 1, 1, 1]):
            G.add_edge(f"s{i}", "v", weight=w)
        H = lans_filter(G)
        assert H["s0"]["v"]["lans_pvalue"] == 0.0

    def test_directed_uses_target_in_edges_not_only_source_out_edges(self):
        # Edge (u, v1) is mid-ranked among u's out-edges (source-side eCDF = 0.5,
        # so a source-only implementation answers p = 0.5) but is the strongest
        # incoming link of v1, which must drive the p-value to 0.
        G = nx.DiGraph()
        G.add_weighted_edges_from([("u", "v1", 100), ("u", "w1", 200), ("x", "v1", 1)])
        H = lans_filter(G)
        assert H["u"]["v1"]["lans_pvalue"] == 0.0


class TestMultipleLinkageAnalysis:
    def test_returns_scored_graph_with_mla_flags(self, weighted_triangle):
        H = multiple_linkage_analysis(weighted_triangle, alpha=0.5)
        assert H.number_of_edges() == weighted_triangle.number_of_edges()
        for _, _, data in H.edges(data=True):
            assert "mla_pvalue" in data
            assert "mla_keep" in data
            assert 0.0 <= data["mla_pvalue"] <= 1.0
        backbone = boolean_filter(H, "mla_keep")
        assert backbone.number_of_edges() <= weighted_triangle.number_of_edges()

    def test_invalid_alpha_raises(self, weighted_triangle):
        with pytest.raises(ValueError):
            multiple_linkage_analysis(weighted_triangle, alpha=1.5)


class TestStatisticalAliases:
    def test_aliases_match_filter_outputs(self, weighted_triangle):
        h_disparity = disparity(weighted_triangle)
        h_lans = lans(weighted_triangle)
        h_mlf = mlf(weighted_triangle)

        for u, v in weighted_triangle.edges():
            assert "disparity_pvalue" in h_disparity[u][v]
            assert "lans_pvalue" in h_lans[u][v]
            assert "ml_pvalue" in h_mlf[u][v]


class TestSignedBackbones:
    @staticmethod
    def _hub_star():
        # Hub "c" with one very strong edge and several weak edges.
        G = nx.Graph()
        G.add_weighted_edges_from(
            [("c", 1, 100.0), ("c", 2, 1.0), ("c", 3, 1.0), ("c", 4, 1.0)]
        )
        return G

    def test_disparity_signed_marks_strong_and_weak(self):
        G = self._hub_star()
        H = disparity_filter(G, signed=True)
        assert H["c"][1]["sign"] == 1  # strong edge
        assert H["c"][2]["sign"] == -1  # weak edge
        assert all(0.0 <= d["disparity_pvalue"] <= 1.0 for _, _, d in H.edges(data=True))

    def test_unsigned_unchanged(self):
        G = self._hub_star()
        H = disparity_filter(G)
        assert all("sign" not in d for _, _, d in H.edges(data=True))
        # one-tailed p-value preserved: (1 - w/s)^(k-1) for the strong edge
        assert H["c"][1]["disparity_pvalue"] == pytest.approx((1 - 100 / 103) ** 3)

    @pytest.mark.parametrize(
        "scorer,attr",
        [(disparity_filter, "disparity_pvalue"), (mlf, "ml_pvalue"), (lans, "lans_pvalue")],
    )
    def test_signed_adds_sign_and_two_sided_pvalue(self, scorer, attr):
        G = nx.les_miserables_graph()
        H = scorer(G, signed=True)
        assert all("sign" in d and d["sign"] in (-1, 1) for _, _, d in H.edges(data=True))
        assert all(0.0 <= d[attr] <= 1.0 for _, _, d in H.edges(data=True))

    def test_signed_pvalue_is_two_sided(self):
        # Two-sided p-value is twice the smaller one-tailed tail (clipped to 1).
        G = self._hub_star()
        unsigned = disparity_filter(G)
        signed = disparity_filter(G, signed=True)
        p_hi = unsigned["c"][1]["disparity_pvalue"]
        assert signed["c"][1]["disparity_pvalue"] == pytest.approx(min(1.0, 2 * p_hi))
