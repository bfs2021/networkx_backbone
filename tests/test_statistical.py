"""Known-answer tests for the statistical scorers (networkx_backbone.statistical).

Each check is tied to an equation in the cited paper, so a regression here
means the implementation drifted from the method it claims to implement.
Shared behaviour (attributes, p-value ranges, validation, multigraphs, empty
graphs, input order) is covered by ``test_api_contract.py``.
"""

import math
import warnings

import networkx as nx
import pytest

import networkx_backbone as nb

pytest.importorskip("scipy")
from scipy import stats


@pytest.fixture
def count_graph():
    G = nx.Graph()
    G.add_weighted_edges_from(
        [("a", "b", 5), ("a", "c", 1), ("b", "c", 2), ("c", "d", 3), ("b", "d", 1)]
    )
    return G


@pytest.fixture
def hub_star():
    """Hub "c" with one very strong edge and several weak edges."""
    G = nx.Graph()
    G.add_weighted_edges_from(
        [("c", 1, 100.0), ("c", 2, 1.0), ("c", 3, 1.0), ("c", 4, 1.0)]
    )
    return G


# ---------------------------------------------------------------------------
# Disparity filter -- Serrano et al. (2009)
# ---------------------------------------------------------------------------


class TestDisparity:
    def test_formula(self, count_graph):
        H = nb.disparity_filter(count_graph)
        # Edge a-b: from a, p = 5/6, k = 2 -> (1 - 5/6)^1; from b, p = 5/8, k=3.
        expected = min((1 - 5 / 6) ** 1, (1 - 5 / 8) ** 2)
        assert H["a"]["b"]["disparity_pvalue"] == pytest.approx(expected)

    def test_unsigned_strong_edge(self, hub_star):
        H = nb.disparity_filter(hub_star)
        assert all("sign" not in d for _, _, d in H.edges(data=True))
        # one-tailed p-value: (1 - w/s)^(k-1) for the strong edge
        assert H["c"][1]["disparity_pvalue"] == pytest.approx((1 - 100 / 103) ** 3)

    def test_directed_uses_source_and_target(self):
        # Five sources each send one link into v; one of them is heavy.  From
        # each source (out-degree 1) nothing is testable, but from v's
        # incoming links the heavy one stands out.
        G = nx.DiGraph()
        G.add_weighted_edges_from(
            [(f"s{i}", "v", w) for i, w in enumerate([100, 1, 1, 1, 1])]
        )
        H = nb.disparity_filter(G)
        expected = (1 - 100 / 104) ** 4
        assert H["s0"]["v"]["disparity_pvalue"] == pytest.approx(expected)
        assert H["s1"]["v"]["disparity_pvalue"] > 0.5

    def test_signed_marks_strong_and_weak(self, hub_star):
        H = nb.disparity_filter(hub_star, signed=True)
        assert H["c"][1]["sign"] == 1  # strong edge
        assert H["c"][2]["sign"] == -1  # weak edge

    def test_signed_pvalue_is_two_sided(self, hub_star):
        # Two-sided p-value is twice the smaller one-tailed tail (clipped to 1).
        unsigned = nb.disparity_filter(hub_star)
        signed = nb.disparity_filter(hub_star, signed=True)
        p_hi = unsigned["c"][1]["disparity_pvalue"]
        assert signed["c"][1]["disparity_pvalue"] == pytest.approx(min(1.0, 2 * p_hi))


@pytest.mark.parametrize(
    ("scorer", "attr"),
    [
        (nb.disparity_filter, "disparity_pvalue"),
        (nb.mlf, "ml_pvalue"),
        (nb.lans, "lans_pvalue"),
    ],
)
def test_signed_adds_sign_and_two_sided_pvalue(scorer, attr, les_miserables_weighted):
    H = scorer(les_miserables_weighted, signed=True)
    assert all("sign" in d and d["sign"] in (-1, 1) for _, _, d in H.edges(data=True))
    assert all(0.0 <= d[attr] <= 1.0 for _, _, d in H.edges(data=True))


def test_aliases_match_full_names(les_miserables_weighted):
    G = les_miserables_weighted
    for alias, full, attr in [
        (nb.disparity, nb.disparity_filter, "disparity_pvalue"),
        (nb.mlf, nb.marginal_likelihood_filter, "ml_pvalue"),
        (nb.lans, nb.lans_filter, "lans_pvalue"),
    ]:
        for signed in (False, True):
            got = nx.get_edge_attributes(alias(G, signed=signed), attr)
            assert got == nx.get_edge_attributes(full(G, signed=signed), attr), (
                alias.__name__
            )


# ---------------------------------------------------------------------------
# Noise-corrected filter -- Coscia & Neffke (2017)
# ---------------------------------------------------------------------------


def test_noise_corrected_formula(count_graph):
    H = nb.noise_corrected_filter(count_graph)
    s = dict(count_graph.degree(weight="weight"))
    W = sum(d["weight"] for *_, d in count_graph.edges(data=True))
    p = s["a"] * s["b"] / W**2
    z = (5 - W * p) / math.sqrt(W * p * (1 - p))
    assert H["a"]["b"]["nc_score"] == pytest.approx(z)


# ---------------------------------------------------------------------------
# Marginal likelihood filter -- Dianati (2016), Eqs. 1-2
# ---------------------------------------------------------------------------


class TestMarginalLikelihood:
    def test_undirected_formula(self, count_graph):
        H = nb.marginal_likelihood_filter(count_graph)
        strength = dict(count_graph.degree(weight="weight"))
        T = sum(strength.values()) / 2
        for u, v, d in H.edges(data=True):
            p = strength[u] * strength[v] / (2 * T * T)
            expected = stats.binom.sf(count_graph[u][v]["weight"] - 1, T, p)
            assert d["ml_pvalue"] == pytest.approx(expected)
        p = nx.get_edge_attributes(H, "ml_pvalue")
        assert min(p, key=p.get) == ("a", "b")  # strongest edge most significant

    def test_directed_formula(self):
        G = nx.DiGraph()
        G.add_weighted_edges_from([(0, 1, 4), (1, 2, 1), (2, 0, 2), (0, 2, 3)])
        H = nb.marginal_likelihood_filter(G)
        s_out = dict(G.out_degree(weight="weight"))
        s_in = dict(G.in_degree(weight="weight"))
        T = sum(s_out.values())
        for u, v, d in H.edges(data=True):
            p = s_out[u] * s_in[v] / T**2
            expected = stats.binom.sf(G[u][v]["weight"] - 1, T, p)
            assert d["ml_pvalue"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# ECM filter -- Gemmetto, Cardillo & Garlaschelli (2017)
# ---------------------------------------------------------------------------


def _ecm_expectations(fit, directed):
    import numpy as np

    a_out, b_out, a_in, b_in, act_out, act_in, _ = fit
    n = len(a_out)
    xxyy = np.exp(-(a_out[:, None] + a_in[None, :]))
    yy = np.exp(-(b_out[:, None] + b_in[None, :]))
    p = xxyy / (1 - yy + xxyy)
    p[~(act_out[:, None] & act_in[None, :])] = 0.0
    p[np.eye(n, dtype=bool)] = 0.0
    w = p / (1 - yy)
    return p, w


class TestECM:
    @pytest.mark.parametrize("directed", [False, True])
    def test_fit_reproduces_degrees_and_strengths(
        self, les_miserables_weighted, directed
    ):
        import numpy as np

        from networkx_backbone.statistical import _fit_ecm

        G = nx.DiGraph(les_miserables_weighted) if directed else les_miserables_weighted
        nodes = list(G)
        k_out = np.array(
            [G.out_degree(v) if directed else G.degree(v) for v in nodes], float
        )
        s_out = np.array(
            [
                G.out_degree(v, weight="weight")
                if directed
                else G.degree(v, weight="weight")
                for v in nodes
            ],
            float,
        )
        k_in = np.array([G.in_degree(v) for v in nodes], float) if directed else k_out
        s_in = (
            np.array([G.in_degree(v, weight="weight") for v in nodes], float)
            if directed
            else s_out
        )
        fit = _fit_ecm(k_out, s_out, k_in, s_in, directed, 1000, 1e-8)
        assert fit[-1] < 1e-6
        p, w = _ecm_expectations(fit, directed)
        np.testing.assert_allclose(p.sum(1), k_out, rtol=1e-6)
        np.testing.assert_allclose(w.sum(1), s_out, rtol=1e-6)
        np.testing.assert_allclose(p.sum(0), k_in, rtol=1e-6)
        np.testing.assert_allclose(w.sum(0), s_in, rtol=1e-6)

    def test_pvalue_is_geometric_tail(self, count_graph):
        # p-value = p_ij * (y_i y_j)^(w - 1): heavier links are rarer.
        H = nb.ecm_filter(count_graph)
        G2 = count_graph.copy()
        G2["a"]["b"]["weight"] = 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            H2 = nb.ecm_filter(G2)
        assert H["a"]["b"]["ecm_pvalue"] < H2["a"]["b"]["ecm_pvalue"]

    def test_pvalues_strictly_positive(self, les_miserables_weighted):
        # The pinned Les Misérables count is in test_les_miserables.py.
        H = nb.ecm_filter(les_miserables_weighted)
        assert all(0 < p <= 1 for *_, p in H.edges(data="ecm_pvalue"))

    def test_warns_when_solver_does_not_converge(self, les_miserables_weighted):
        with pytest.warns(RuntimeWarning, match="did not converge"):
            nb.ecm_filter(les_miserables_weighted, max_iter=1)

    def test_self_loop_gets_pvalue_one(self, count_graph):
        count_graph.add_edge("a", "a", weight=4)
        H = nb.ecm_filter(count_graph)
        assert H["a"]["a"]["ecm_pvalue"] == 1.0


# ---------------------------------------------------------------------------
# LANS -- Foti, Hughes & Rockmore (2011); multiple linkage analysis
# ---------------------------------------------------------------------------


class TestLANS:
    def test_empirical_cdf(self, count_graph):
        H = nb.lans_filter(count_graph)
        # Edge c-d: at c weights are [1, 2, 3] -> F(3) = 1; at d [1, 3] -> 1.
        assert H["c"]["d"]["lans_pvalue"] == 0.0
        # Edge a-c: at a [1, 5] -> F(1) = 1/2; at c F(1) = 1/3 -> 1 - 1/2.
        assert H["a"]["c"]["lans_pvalue"] == pytest.approx(0.5)

    def test_strongest_link_of_each_node_is_kept(self, les_miserables_weighted):
        H = nb.lans_filter(les_miserables_weighted)
        for node in H:
            if H.degree(node) == 0:
                continue
            best = max(H[node], key=lambda v: H[node][v]["weight"])
            assert H[node][best]["lans_pvalue"] == 0.0

    def test_mla_invalid_alpha_raises(self, weighted_triangle):
        with pytest.raises(ValueError):
            nb.multiple_linkage_analysis(weighted_triangle, alpha=1.5)
