"""Known-answer and edge-case tests for the statistical scorers.

Each check is tied to an equation in the cited paper, so a regression here
means the implementation drifted from the method it claims to implement.
"""

import math
import warnings

import networkx as nx
import pytest

import networkx_backbone as nb

pytest.importorskip("scipy")
from scipy import stats

SCORERS = [
    (nb.disparity_filter, "disparity_pvalue"),
    (nb.noise_corrected_filter, "nc_score"),
    (nb.marginal_likelihood_filter, "ml_pvalue"),
    (nb.ecm_filter, "ecm_pvalue"),
    (nb.lans_filter, "lans_pvalue"),
    (nb.multiple_linkage_analysis, "mla_pvalue"),
]


@pytest.fixture
def count_graph():
    G = nx.Graph()
    G.add_weighted_edges_from(
        [("a", "b", 5), ("a", "c", 1), ("b", "c", 2), ("c", "d", 3), ("b", "d", 1)]
    )
    return G


# ---------------------------------------------------------------------------
# Shared conventions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("scorer", "attr"), SCORERS)
def test_rejects_multigraph(scorer, attr):
    G = nx.MultiGraph()
    G.add_edge(0, 1, weight=1.0)
    G.add_edge(0, 1, weight=2.0)
    with pytest.raises(nx.NetworkXNotImplemented):
        scorer(G)


@pytest.mark.parametrize(("scorer", "attr"), SCORERS)
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0, -1.0])
def test_rejects_invalid_weights(scorer, attr, bad):
    G = nx.path_graph(4)
    nx.set_edge_attributes(G, 1.0, "weight")
    G[1][2]["weight"] = bad
    with pytest.raises(nx.NetworkXError):
        scorer(G)


@pytest.mark.parametrize(("scorer", "attr"), SCORERS)
def test_does_not_mutate_input(scorer, attr, count_graph):
    before = {(u, v): dict(d) for u, v, d in count_graph.edges(data=True)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        scorer(count_graph)
    assert {(u, v): d for u, v, d in count_graph.edges(data=True)} == before


@pytest.mark.parametrize(("scorer", "attr"), SCORERS)
def test_independent_of_node_insertion_order(scorer, attr, les_miserables_weighted):
    G = les_miserables_weighted
    R = nx.Graph()
    R.add_nodes_from(reversed(list(G)))
    R.add_edges_from((v, u, d) for u, v, d in reversed(list(G.edges(data=True))))
    a = nx.get_edge_attributes(scorer(G), attr)
    b = nx.get_edge_attributes(scorer(R), attr)
    for (u, v), val in a.items():
        other = b[(u, v)] if (u, v) in b else b[(v, u)]
        assert other == pytest.approx(val, rel=1e-6, abs=1e-9)


@pytest.mark.parametrize(("scorer", "attr"), SCORERS)
def test_empty_graph(scorer, attr):
    H = scorer(nx.Graph())
    assert H.number_of_nodes() == 0


# ---------------------------------------------------------------------------
# Disparity filter -- Serrano et al. (2009)
# ---------------------------------------------------------------------------


class TestDisparityReference:
    def test_formula(self, count_graph):
        H = nb.disparity_filter(count_graph)
        # Edge a-b: from a, p = 5/6, k = 2 -> (1 - 5/6)^1; from b, p = 5/8, k=3.
        expected = min((1 - 5 / 6) ** 1, (1 - 5 / 8) ** 2)
        assert H["a"]["b"]["disparity_pvalue"] == pytest.approx(expected)

    def test_directed_uses_source_and_target(self):
        # Five sources each send one link into v; one of them is heavy.  From
        # each source (out-degree 1) nothing is testable, but from v's
        # incoming links the heavy one stands out.
        G = nx.DiGraph()
        G.add_weighted_edges_from([(f"s{i}", "v", w) for i, w in enumerate([100, 1, 1, 1, 1])])
        H = nb.disparity_filter(G)
        expected = (1 - 100 / 104) ** 4
        assert H["s0"]["v"]["disparity_pvalue"] == pytest.approx(expected)
        assert H["s1"]["v"]["disparity_pvalue"] > 0.5


# ---------------------------------------------------------------------------
# Marginal likelihood filter -- Dianati (2016), Eqs. 1-2
# ---------------------------------------------------------------------------


class TestMarginalLikelihoodReference:
    def test_undirected_formula(self, count_graph):
        H = nb.marginal_likelihood_filter(count_graph)
        strength = dict(count_graph.degree(weight="weight"))
        T = sum(strength.values()) / 2
        for u, v, d in H.edges(data=True):
            p = strength[u] * strength[v] / (2 * T * T)
            expected = stats.binom.sf(count_graph[u][v]["weight"] - 1, T, p)
            assert d["ml_pvalue"] == pytest.approx(expected)

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

    def test_strongest_edge_is_most_significant(self, count_graph):
        H = nb.marginal_likelihood_filter(count_graph)
        p = nx.get_edge_attributes(H, "ml_pvalue")
        assert min(p, key=p.get) == ("a", "b")

    def test_les_miserables_reference(self, les_miserables_weighted):
        H = nb.marginal_likelihood_filter(les_miserables_weighted)
        kept = sum(d["ml_pvalue"] < 0.05 for *_, d in H.edges(data=True))
        assert kept == 126


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


class TestECMReference:
    @pytest.mark.parametrize("directed", [False, True])
    def test_fit_reproduces_degrees_and_strengths(self, les_miserables_weighted, directed):
        import numpy as np

        from networkx_backbone.statistical import _fit_ecm

        G = nx.DiGraph(les_miserables_weighted) if directed else les_miserables_weighted
        nodes = list(G)
        k_out = np.array([G.out_degree(v) if directed else G.degree(v) for v in nodes], float)
        s_out = np.array(
            [
                G.out_degree(v, weight="weight") if directed else G.degree(v, weight="weight")
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

    def test_filters_les_miserables(self, les_miserables_weighted):
        H = nb.ecm_filter(les_miserables_weighted)
        p = [d["ecm_pvalue"] for *_, d in H.edges(data=True)]
        assert all(0 < x <= 1 for x in p)
        assert sum(x < 0.05 for x in p) == 72

    def test_warns_when_solver_does_not_converge(self, les_miserables_weighted):
        with pytest.warns(RuntimeWarning, match="did not converge"):
            nb.ecm_filter(les_miserables_weighted, max_iter=1)

    def test_self_loop_gets_pvalue_one(self, count_graph):
        count_graph.add_edge("a", "a", weight=4)
        H = nb.ecm_filter(count_graph)
        assert H["a"]["a"]["ecm_pvalue"] == 1.0


# ---------------------------------------------------------------------------
# LANS -- Foti, Hughes & Rockmore (2011)
# ---------------------------------------------------------------------------


class TestLANSReference:
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


def test_noise_corrected_formula(count_graph):
    H = nb.noise_corrected_filter(count_graph)
    s = dict(count_graph.degree(weight="weight"))
    W = sum(d["weight"] for *_, d in count_graph.edges(data=True))
    p = s["a"] * s["b"] / W**2
    z = (5 - W * p) / math.sqrt(W * p * (1 - p))
    assert H["a"]["b"]["nc_score"] == pytest.approx(z)
