"""Tests for networkx_backbone.bipartite: projections, null models, wrappers.

Shared behaviour of every projection scorer (attributes, p-value ranges,
agent-set validation, directed/multigraph rejection, generator input, empty
graphs, mixed labels, no input mutation) is covered by ``test_api_contract.py``.
"""

import itertools
import warnings

import networkx as nx
import numpy as np
import pytest
from scipy import stats

import networkx_backbone as nb
from networkx_backbone import (
    backbone,
    backbone_from_projection,
    backbone_from_unweighted,
    backbone_from_weighted,
    bipartite_projection,
    fastball,
    fdsm,
    hyper_projection,
    probs_projection,
    sdsm,
    simple_projection,
    ycn_projection,
)
from networkx_backbone import bipartite as bp

AGENTS = [1, 2, 3]


def _davis_matrix(B, women):
    agents, artifacts, R, observed = bp._bipartite_projection_matrix(B, women)
    return agents, artifacts, R, observed


def _brute_pb_pmf(p):
    """Poisson-binomial pmf by enumerating all outcomes."""
    pmf = np.zeros(len(p) + 1)
    for bits in itertools.product([0, 1], repeat=len(p)):
        pmf[sum(bits)] += np.prod([q if b else 1 - q for q, b in zip(p, bits)])
    return pmf


def _weights(G):
    return {frozenset((u, v)): w for u, v, w in G.edges(data="weight")}


# ---------------------------------------------------------------------------
# Projections
# ---------------------------------------------------------------------------


def test_davis_southern_women_fixture(davis_southern_women_graph, davis_women_nodes):
    assert nx.is_bipartite(davis_southern_women_graph)
    assert davis_southern_women_graph.number_of_nodes() == 32
    assert davis_southern_women_graph.number_of_edges() == 89
    assert len(davis_women_nodes) == 18


@pytest.mark.parametrize(
    ("func", "method", "weight"),
    [
        (simple_projection, "simple", 1.0),
        (hyper_projection, "hyper", 0.5),
        (probs_projection, "probs", 0.25),
        (ycn_projection, "ycn", 1.0 / 12.0),
    ],
)
def test_projection_weights(bipartite_graph, func, method, weight):
    projection = func(bipartite_graph, agent_nodes=AGENTS)
    assert projection.number_of_nodes() == 3
    assert projection.number_of_edges() == 3
    for _, _, data in projection.edges(data=True):
        assert data["weight"] == pytest.approx(weight)
    dispatched = bipartite_projection(bipartite_graph, AGENTS, method=method)
    assert _weights(dispatched) == _weights(projection)


def test_probs_projection_directed():
    B = nx.Graph()
    B.add_edges_from([("a", "x"), ("a", "y"), ("b", "x"), ("c", "y")])
    projection = probs_projection(B, agent_nodes=["a", "b", "c"], directed=True)
    assert isinstance(projection, nx.DiGraph)
    assert projection["a"]["b"]["weight"] == pytest.approx(0.25)
    assert projection["b"]["a"]["weight"] == pytest.approx(0.5)


def test_bipartite_projection_dispatch_unknown_method_raises(bipartite_graph):
    with pytest.raises(ValueError):
        bipartite_projection(bipartite_graph, agent_nodes=AGENTS, method="unknown")


def test_scorers_annotate_projection_weights(
    davis_southern_women_graph, davis_women_nodes
):
    B, women = davis_southern_women_graph, davis_women_nodes
    for scored, method in [
        (sdsm(B, women, projection="hyper"), "hyper"),
        (fdsm(B, women, trials=20, seed=42, projection="ycn"), "ycn"),
        (backbone_from_projection(B, women, "fixedrow", 0.5, "probs"), "probs"),
    ]:
        expected = _weights(bipartite_projection(B, women, method=method))
        got = _weights(scored)
        assert got, method
        assert got == {e: pytest.approx(expected.get(e, 0.0)) for e in got}, method


def test_invalid_projection_raises(bipartite_graph):
    with pytest.raises(ValueError):
        sdsm(bipartite_graph, agent_nodes=AGENTS, projection="invalid")


def test_validation_errors(bipartite_graph):
    with pytest.raises(nx.NetworkXError, match="not bipartite"):
        sdsm(nx.complete_graph(3), agent_nodes=[0, 1])
    with pytest.raises(nx.NetworkXError, match="not in B"):
        sdsm(bipartite_graph, agent_nodes=[99])


# ---------------------------------------------------------------------------
# BiCM
# ---------------------------------------------------------------------------


class TestBiCM:
    def test_margins_match_observed_degrees(
        self, davis_southern_women_graph, davis_women_nodes
    ):
        B = davis_southern_women_graph
        P, agents, artifacts = nb.bicm(B, davis_women_nodes, return_labels=True)
        row = np.array([B.degree(a) for a in agents])
        col = np.array([B.degree(f) for f in artifacts])
        np.testing.assert_allclose(P.sum(axis=1), row, atol=1e-6)
        np.testing.assert_allclose(P.sum(axis=0), col, atol=1e-6)
        assert np.all((P > 0) & (P < 1))

    def test_is_maximum_entropy_form(
        self, davis_southern_women_graph, davis_women_nodes
    ):
        # p_ik = x_i y_k / (1 + x_i y_k)  <=>  logit(P) has rank one in the
        # additive sense: logit(p_ik) = a_i + b_k.
        P = nb.bicm(davis_southern_women_graph, davis_women_nodes)
        L = np.log(P / (1 - P))
        resid = L - L.mean(1, keepdims=True) - L.mean(0, keepdims=True) + L.mean()
        assert np.abs(resid).max() < 1e-8

    def test_degenerate_rows_and_columns_fixed(self):
        # agent "full" links every artifact, "iso" links none; artifact "z"
        # is linked only by "full" (column degree 1).
        B = nx.Graph()
        B.add_edges_from([("full", f) for f in "xyz"])
        B.add_edges_from([("a", "x"), ("b", "y"), ("a", "y")])
        B.add_node("iso")
        P, agents, artifacts = nb.bicm(B, ["full", "iso", "a", "b"], return_labels=True)
        ai = {v: i for i, v in enumerate(agents)}
        fi = {v: i for i, v in enumerate(artifacts)}
        np.testing.assert_array_equal(P[ai["full"]], 1.0)
        np.testing.assert_array_equal(P[ai["iso"]], 0.0)
        assert P[ai["a"], fi["z"]] == 0.0 and P[ai["b"], fi["z"]] == 0.0
        R = np.array([[int(B.has_edge(a, f)) for f in artifacts] for a in agents])
        np.testing.assert_allclose(P.sum(axis=1), R.sum(axis=1), atol=1e-6)
        np.testing.assert_allclose(P.sum(axis=0), R.sum(axis=0), atol=1e-6)

    def test_non_convergence_warns(self, davis_southern_women_graph, davis_women_nodes):
        with pytest.warns(RuntimeWarning, match="did not converge"):
            nb.bicm(davis_southern_women_graph, davis_women_nodes, max_iter=1)

    def test_converged_fit_does_not_warn(
        self, davis_southern_women_graph, davis_women_nodes
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            nb.bicm(davis_southern_women_graph, davis_women_nodes)


def test_fastball_preserves_marginals():
    m = np.array(
        [
            [1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [1, 1, 0, 0, 1],
            [0, 0, 1, 1, 1],
        ],
        dtype=int,
    )
    randomized = fastball(m, n_swaps=50, seed=42)
    assert randomized.shape == m.shape
    assert np.array_equal(randomized.sum(axis=1), m.sum(axis=1))
    assert np.array_equal(randomized.sum(axis=0), m.sum(axis=0))


# ---------------------------------------------------------------------------
# Poisson-binomial helpers and SDSM p-values
# ---------------------------------------------------------------------------


class TestPoissonBinomial:
    def test_exact_matches_brute_force(self):
        for seed in range(4):
            p = np.random.default_rng(seed).random(9)
            pmf = _brute_pb_pmf(p)
            np.testing.assert_allclose(bp._poisson_binomial_pmf(p), pmf, atol=1e-13)
            for k in range(-1, 11):
                expected_sf = pmf[max(k, 0) :].sum()
                expected_cdf = pmf[: max(k + 1, 0)].sum()
                assert bp._poisson_binomial_sf(p, k) == pytest.approx(
                    expected_sf, abs=1e-12
                ), (seed, k)
                assert bp._poisson_binomial_cdf(p, k) == pytest.approx(
                    expected_cdf, abs=1e-12
                ), (seed, k)

    def test_rna_formula(self):
        p = np.random.default_rng(1).random(30) * 0.6
        mu, sigma = p.sum(), np.sqrt((p * (1 - p)).sum())
        gamma = (p * (1 - p) * (1 - 2 * p)).sum() / sigma**3

        def G(k):
            x = (k + 0.5 - mu) / sigma
            val = stats.norm.cdf(x) + gamma * (1 - x**2) * stats.norm.pdf(x) / 6
            return min(max(val, 0.0), 1.0)

        for k in range(20):
            upper, lower = bp._rna_tails(mu, (p**2).sum(), (p**3).sum(), np.array([k]))
            assert upper[0] == pytest.approx(1 - G(k - 1) if k > 0 else 1.0, abs=1e-12)
            assert lower[0] == pytest.approx(G(k), abs=1e-12)

    def test_rna_close_to_exact(self, davis_southern_women_graph, davis_women_nodes):
        _, _, R, obs = _davis_matrix(davis_southern_women_graph, davis_women_nodes)
        P = bp._bicm_fit(R)
        k = obs[np.triu_indices(len(R), 1)]
        exact, _ = bp._pairwise_pb_tails(P, k, method="exact")
        rna, _ = bp._pairwise_pb_tails(P, k, method="rna")
        assert np.abs(exact - rna).max() < 0.03

    def test_sdsm_exact_pvalues_match_brute_force(self):
        B = nx.Graph()
        B.add_edges_from(
            [
                ("a", 1),
                ("a", 2),
                ("a", 3),
                ("b", 1),
                ("b", 2),
                ("c", 3),
                ("c", 4),
                ("d", 1),
                ("d", 4),
                ("d", 5),
                ("e", 2),
                ("e", 5),
            ]
        )
        agents = ["a", "b", "c", "d", "e"]
        P, ag, _ = nb.bicm(B, agents, return_labels=True)
        R = np.array([[int(B.has_edge(u, f)) for f in [1, 2, 3, 4, 5]] for u in ag])
        obs = R @ R.T
        for method in ("exact", "auto"):
            for signed in (False, True):
                H = nb.sdsm(B, agents, signed=signed, pvalue_method=method)
                for i, j in itertools.combinations(range(len(ag)), 2):
                    pmf = _brute_pb_pmf(P[i] * P[j])
                    hi, lo = pmf[obs[i, j] :].sum(), pmf[: obs[i, j] + 1].sum()
                    expected = min(1.0, 2 * min(hi, lo)) if signed else hi
                    got = H[ag[i]][ag[j]]["sdsm_pvalue"]
                    assert got == pytest.approx(expected, abs=1e-10)

    def test_invalid_pvalue_method(self, bipartite_graph):
        with pytest.raises(ValueError):
            nb.sdsm(bipartite_graph, [1, 2, 3], pvalue_method="normal")


# ---------------------------------------------------------------------------
# SDSM-EC
# ---------------------------------------------------------------------------


class TestSDSMEC:
    def _cells(self, B, women):
        agents, artifacts, R, _ = _davis_matrix(B, women)
        required = [(agents[0], artifacts[k]) for k in np.flatnonzero(R[0])[:2]]
        prohibited = [
            (agents[i], artifacts[k])
            for i in range(1, 4)
            for k in range(5)
            if R[i, k] == 0
        ]
        return agents, artifacts, R, required, prohibited

    def test_refit_preserves_margins(
        self, davis_southern_women_graph, davis_women_nodes
    ):
        agents, artifacts, R, required, prohibited = self._cells(
            davis_southern_women_graph, davis_women_nodes
        )
        a_idx = {v: i for i, v in enumerate(agents)}
        f_idx = {v: i for i, v in enumerate(artifacts)}
        req = bp._constraint_mask(required, "required", a_idx, f_idx, R.shape)
        pro = bp._constraint_mask(prohibited, "prohibited", a_idx, f_idx, R.shape)
        P = bp._bicm_fit(R, required=req, prohibited=pro)
        np.testing.assert_allclose(P.sum(axis=1), R.sum(axis=1), atol=1e-6)
        np.testing.assert_allclose(P.sum(axis=0), R.sum(axis=0), atol=1e-6)
        assert np.all(P[req] == 1.0) and np.all(P[pro] == 0.0)

    def test_constraints_accept_numpy_array(
        self, davis_southern_women_graph, davis_women_nodes
    ):
        _, _, _, _, prohibited = self._cells(
            davis_southern_women_graph, davis_women_nodes
        )
        as_list = nb.sdsm(
            davis_southern_women_graph, davis_women_nodes, prohibited=prohibited
        )
        as_array = nb.sdsm(
            davis_southern_women_graph,
            davis_women_nodes,
            prohibited=np.array(prohibited),
        )
        for u, v, p in as_list.edges(data="sdsm_pvalue"):
            assert as_array[u][v]["sdsm_pvalue"] == pytest.approx(p)

    def test_reversed_pair_raises(self, davis_southern_women_graph, davis_women_nodes):
        _, _, _, _, prohibited = self._cells(
            davis_southern_women_graph, davis_women_nodes
        )
        a, f = prohibited[0]
        with pytest.raises(nx.NetworkXError, match="ordered"):
            nb.sdsm(davis_southern_women_graph, davis_women_nodes, prohibited=[(f, a)])

    def test_malformed_pairs_raise(self, davis_southern_women_graph, davis_women_nodes):
        for bad in [["abc"], [("x",)], [("a", "b", "c")], [5]]:
            with pytest.raises(nx.NetworkXError, match="pairs"):
                nb.sdsm(davis_southern_women_graph, davis_women_nodes, required=bad)

    def test_unknown_constraint_nodes_raise(
        self, davis_southern_women_graph, davis_women_nodes
    ):
        B = davis_southern_women_graph
        with pytest.raises(nx.NetworkXError):
            sdsm(B, agent_nodes=davis_women_nodes, prohibited=[("missing", "absent")])

    def test_agent_agent_pair_raises(
        self, davis_southern_women_graph, davis_women_nodes
    ):
        pair = (davis_women_nodes[0], davis_women_nodes[1])
        with pytest.raises(nx.NetworkXError):
            nb.sdsm(davis_southern_women_graph, davis_women_nodes, prohibited=[pair])

    def test_contradictory_constraints_raise(
        self, davis_southern_women_graph, davis_women_nodes
    ):
        _, _, _, required, prohibited = self._cells(
            davis_southern_women_graph, davis_women_nodes
        )
        with pytest.raises(nx.NetworkXError):
            nb.sdsm(
                davis_southern_women_graph, davis_women_nodes, prohibited=required[:1]
            )
        with pytest.raises(nx.NetworkXError):
            nb.sdsm(
                davis_southern_women_graph, davis_women_nodes, required=prohibited[:1]
            )

    def test_empty_constraints_equal_unconstrained(
        self, davis_southern_women_graph, davis_women_nodes
    ):
        base = nb.sdsm(davis_southern_women_graph, davis_women_nodes)
        empty = nb.sdsm(
            davis_southern_women_graph,
            davis_women_nodes,
            prohibited=[],
            required=np.empty((0, 2)),
        )
        for u, v, p in base.edges(data="sdsm_pvalue"):
            assert empty[u][v]["sdsm_pvalue"] == pytest.approx(p)


# ---------------------------------------------------------------------------
# FDSM
# ---------------------------------------------------------------------------


class TestFDSMNull:
    def test_samples_preserve_degree_sequence(
        self, davis_southern_women_graph, davis_women_nodes
    ):
        _, _, R, _ = _davis_matrix(davis_southern_women_graph, davis_women_nodes)
        samples = list(bp._fdsm_samples(R, 200, np.random.default_rng(0)))
        assert len(samples) == 200
        for M in samples:
            np.testing.assert_array_equal(M.sum(axis=1), R.sum(axis=1))
            np.testing.assert_array_equal(M.sum(axis=0), R.sum(axis=0))
            assert set(np.unique(M)) <= {0, 1}
        # The chain actually moves.
        assert any(not np.array_equal(M, R) for M in samples)

    @pytest.mark.parametrize("signed", [False, True])
    def test_pvalues_strictly_positive(
        self, davis_southern_women_graph, davis_women_nodes, signed
    ):
        trials = 50
        H = nb.fdsm(
            davis_southern_women_graph,
            davis_women_nodes,
            trials=trials,
            seed=0,
            signed=signed,
        )
        pvals = [p for *_, p in H.edges(data="fdsm_pvalue")]
        assert min(pvals) >= 1 / (trials + 1) - 1e-12
        assert max(pvals) <= 1.0

    def test_pvalue_matches_exact_null_on_tiny_graph(self):
        # 2x2 identity-like matrix: the FDSM space is {I, anti-I}, both with
        # co-occurrence 0, so the upper-tail p-value is exactly 1.
        B = nx.Graph([("a", "x"), ("b", "y"), ("c", "x"), ("c", "y")])
        H = nb.fdsm(B, ["a", "b", "c"], trials=20, seed=1)
        assert H["a"]["b"]["fdsm_pvalue"] == pytest.approx(1.0)

    def test_seed_types_reproducible(
        self, davis_southern_women_graph, davis_women_nodes
    ):
        def pvalues(seed):
            H = nb.fdsm(
                davis_southern_women_graph, davis_women_nodes, trials=30, seed=seed
            )
            return dict(nx.get_edge_attributes(H, "fdsm_pvalue"))

        for make_seed in (np.random.default_rng, np.random.RandomState, int):
            assert pvalues(make_seed(3)) == pvalues(make_seed(3)), make_seed

    def test_invalid_trials_and_burn_in_raise(self, bipartite_graph):
        with pytest.raises(ValueError):
            fdsm(bipartite_graph, agent_nodes=AGENTS, trials=0)
        with pytest.raises(ValueError):
            nb.fdsm(bipartite_graph, AGENTS, trials=5, burn_in=-1)


# ---------------------------------------------------------------------------
# Fixed fill / row / column models
# ---------------------------------------------------------------------------


def _random_bipartite(n_agents, n_artifacts, density, seed):
    rng = np.random.default_rng(seed)
    R = (rng.random((n_agents, n_artifacts)) < density).astype(int)
    B = nx.Graph()
    agents = [f"a{i}" for i in range(n_agents)]
    artifacts = [f"f{k}" for k in range(n_artifacts)]
    B.add_nodes_from(agents)
    B.add_nodes_from(artifacts)
    B.add_edges_from((agents[i], artifacts[k]) for i, k in zip(*np.nonzero(R)))
    return B, agents, artifacts, R


class TestFixedModels:
    @pytest.mark.parametrize(
        "dims", [(3, 3, 4), (4, 3, 7), (3, 4, 5), (2, 4, 6), (4, 4, 3)]
    )
    def test_fixedfill_pmf_matches_enumeration(self, dims):
        na, nf, T = dims
        counts = np.zeros(nf + 1)
        for cells in itertools.combinations(range(na * nf), T):
            M = np.zeros(na * nf, dtype=int)
            M[list(cells)] = 1
            M = M.reshape(na, nf)
            counts[M[0] @ M[1]] += 1
        np.testing.assert_allclose(
            bp._fixedfill_pmf(na, nf, T), counts / counts.sum(), atol=1e-12
        )

    def test_fixedfill_pvalues(self):
        B, agents, _, R = _random_bipartite(3, 4, 0.5, seed=2)
        pmf = bp._fixedfill_pmf(3, 4, int(R.sum()))
        H = nb.fixedfill(B, agents, alpha=np.inf, signed=False)
        obs = R @ R.T
        for i, j in itertools.combinations(range(3), 2):
            assert H[agents[i]][agents[j]]["fixedfill_pvalue"] == pytest.approx(
                pmf[obs[i, j] :].sum()
            )

    def test_fixedcol_matches_enumeration(self):
        # Columns: x={a,b}, y={a,b}, z={a,c,d}, w={b,c}
        B = nx.Graph()
        cols = {"x": "ab", "y": "ab", "z": "acd", "w": "bc"}
        for f, members in cols.items():
            B.add_edges_from((a, f) for a in members)
        agents = ["a", "b", "c", "d"]
        counts = np.zeros(len(cols) + 1)
        combos = [list(itertools.combinations(range(4), len(m))) for m in cols.values()]
        for choice in itertools.product(*combos):
            counts[sum(1 for ch in choice if 0 in ch and 1 in ch)] += 1
        pmf = counts / counts.sum()
        H = nb.fixedcol(B, agents, alpha=np.inf, signed=False)
        assert H["a"]["b"]["fixedcol_pvalue"] == pytest.approx(pmf[2:].sum())
        Hs = nb.fixedcol(B, agents, alpha=np.inf, signed=True)
        assert Hs["a"]["b"]["fixedcol_pvalue"] == pytest.approx(
            min(1.0, 2 * min(pmf[2:].sum(), pmf[:3].sum()))
        )

    def test_fixedrow_matches_hypergeometric(self):
        B, agents, _, R = _random_bipartite(6, 8, 0.4, seed=5)
        H = nb.fixedrow(B, agents, alpha=np.inf)
        obs = R @ R.T
        d = R.sum(axis=1)
        for i, j in itertools.combinations(range(6), 2):
            expected = stats.hypergeom.sf(obs[i, j] - 1, 8, d[i], d[j])
            assert H[agents[i]][agents[j]]["fixedrow_pvalue"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Agent-set inference
# ---------------------------------------------------------------------------


class TestAgentInference:
    def test_disconnected_inference_ambiguous(self):
        B = nx.Graph([("a", "x"), ("b", "x"), ("c", "y"), ("d", "y")])
        with pytest.raises(nx.AmbiguousSolution):
            nb.backbone_from_projection(B, method="fixedrow")

    def test_inference_uses_bipartite_attribute(self):
        B = nx.Graph([("a", "x"), ("b", "x"), ("c", "y"), ("d", "y")])
        nx.set_node_attributes(B, {n: 0 if n in "abcd" else 1 for n in B}, "bipartite")
        H = nb.backbone_from_projection(B, method="fixedrow", alpha=1.0)
        assert set(H.nodes()) == {"a", "b", "c", "d"}

    def test_inference_connected_graph(self, bipartite_graph):
        H = nb.backbone_from_projection(bipartite_graph, method="fixedrow", alpha=1.0)
        assert set(H.nodes()) in ({1, 2, 3}, {"a", "b", "c"})


# ---------------------------------------------------------------------------
# Wrapper thresholding
# ---------------------------------------------------------------------------


class TestWrapperThresholding:
    @pytest.mark.parametrize(
        "method", ["sdsm", "fdsm", "fixedfill", "fixedrow", "fixedcol"]
    )
    @pytest.mark.parametrize("signed", [False, True])
    def test_threshold_consistent_across_methods(
        self, davis_southern_women_graph, davis_women_nodes, method, signed
    ):
        kw = {"trials": 100, "seed": 4} if method == "fdsm" else {}
        alpha = 0.2
        H = nb.backbone_from_projection(
            davis_southern_women_graph,
            davis_women_nodes,
            method=method,
            alpha=alpha,
            signed=signed,
            **kw,
        )
        attr = f"{method}_pvalue"
        assert all(d[attr] < alpha for *_, d in H.edges(data=True))
        if signed:
            assert all(d["sign"] in (-1, 1) for *_, d in H.edges(data=True))
        # Same set as scoring every pair and thresholding by hand.
        scored = getattr(nb, method)(
            davis_southern_women_graph,
            davis_women_nodes,
            alpha=np.inf,
            signed=signed,
            **kw,
        )
        expected = {
            frozenset((u, v)) for u, v, p in scored.edges(data=attr) if p < alpha
        }
        assert {frozenset(e) for e in H.edges()} == expected
        assert (
            H.number_of_edges()
            < len(davis_women_nodes) * (len(davis_women_nodes) - 1) // 2
        )

    @pytest.mark.parametrize("method", ["sdsm", "fixedrow"])
    def test_mtc_forwarded(self, davis_southern_women_graph, davis_women_nodes, method):
        alpha = 0.3
        scored = getattr(nb, method)(
            davis_southern_women_graph, davis_women_nodes, alpha=np.inf
        )
        attr = f"{method}_pvalue"
        edges = list(scored.edges(data=attr))
        adjusted = nb.adjust_pvalues([p for *_, p in edges], "bonferroni")
        expected = {
            frozenset((u, v)) for (u, v, _), q in zip(edges, adjusted) if q < alpha
        }
        H = nb.backbone_from_projection(
            davis_southern_women_graph,
            davis_women_nodes,
            method=method,
            alpha=alpha,
            mtc="bonferroni",
        )
        assert {frozenset(e) for e in H.edges()} == expected

    def test_backbone_auto_dispatch_thresholds(self, davis_southern_women_graph):
        H = nb.backbone(davis_southern_women_graph, alpha=0.1)
        assert all(d["sdsm_pvalue"] < 0.1 for *_, d in H.edges(data=True))
        assert H.number_of_edges() < 153


# ---------------------------------------------------------------------------
# High-level wrappers
# ---------------------------------------------------------------------------


def test_backbone_from_projection_sdsm_with_custom_projection(
    davis_southern_women_graph, davis_women_nodes
):
    backbone_graph = backbone_from_projection(
        davis_southern_women_graph,
        davis_women_nodes,
        method="sdsm",
        alpha=0.5,
        projection="ycn",
    )
    scored = sdsm(davis_southern_women_graph, davis_women_nodes, projection="ycn")
    expected = {
        frozenset((u, v)) for u, v, p in scored.edges(data="sdsm_pvalue") if p < 0.5
    }
    assert backbone_graph.number_of_nodes() == len(davis_women_nodes)
    assert {frozenset(e) for e in backbone_graph.edges()} == expected
    assert expected
    for _, _, data in backbone_graph.edges(data=True):
        assert data["sdsm_pvalue"] < 0.5
        assert "weight" in data


def test_backbone_from_weighted_dispatch(weighted_triangle):
    backbone_graph = backbone_from_weighted(
        weighted_triangle, method="disparity", alpha=1.0
    )
    assert backbone_graph.number_of_nodes() == weighted_triangle.number_of_nodes()
    with pytest.raises(ValueError):
        backbone_from_weighted(weighted_triangle, method="global")


def test_backbone_from_weighted_multigraph_auto_collapse():
    G = nx.MultiGraph()
    G.add_edge(0, 1, edge_type="a")
    G.add_edge(0, 1, edge_type="b")
    G.add_edge(0, 1, edge_type="b")
    G.add_edge(1, 2, edge_type="a")
    backbone_graph = backbone_from_weighted(
        G, method="global", threshold=2, edge_type_attr="edge_type"
    )
    assert backbone_graph.number_of_edges() == 1
    assert backbone_graph.has_edge(0, 1)


def test_backbone_from_weighted_multidigraph_auto_collapse():
    G = nx.MultiDiGraph()
    G.add_edge(0, 1)
    G.add_edge(0, 1)
    G.add_edge(1, 0)
    backbone_graph = backbone_from_weighted(G, method="global", threshold=2)
    assert isinstance(backbone_graph, nx.DiGraph)
    assert list(backbone_graph.edges()) == [(0, 1)]


def test_backbone_from_unweighted_dispatch(karate):
    backbone_graph = backbone_from_unweighted(karate, method="lspar", s=0.5)
    assert backbone_graph.number_of_nodes() == karate.number_of_nodes()


def test_backbone_dispatch(
    davis_southern_women_graph, davis_women_nodes, weighted_triangle
):
    proj_backbone = backbone(
        davis_southern_women_graph,
        method="sdsm",
        agent_nodes=davis_women_nodes,
        alpha=0.5,
    )
    weighted_backbone = backbone(weighted_triangle, method="disparity", alpha=1.0)
    assert set(proj_backbone.nodes()).issubset(set(davis_women_nodes))
    assert weighted_backbone.number_of_nodes() == weighted_triangle.number_of_nodes()

    with pytest.raises(ValueError):
        backbone(weighted_triangle, method="not_a_method")


class TestSignedProjections:
    # Signed fixed models and the wrapper are covered by TestWrapperThresholding.

    def test_sdsm_signed_adds_sign(self, davis_southern_women_graph, davis_women_nodes):
        H = sdsm(davis_southern_women_graph, agent_nodes=davis_women_nodes, signed=True)
        assert {d["sign"] for _, _, d in H.edges(data=True)} == {-1, 1}
        assert all(0.0 <= d["sdsm_pvalue"] <= 1.0 for _, _, d in H.edges(data=True))
        H = sdsm(davis_southern_women_graph, agent_nodes=davis_women_nodes)
        assert all("sign" not in d for _, _, d in H.edges(data=True))

    def test_fdsm_signed_deterministic(
        self, davis_southern_women_graph, davis_women_nodes
    ):
        kw = {
            "agent_nodes": davis_women_nodes,
            "trials": 200,
            "seed": 7,
            "signed": True,
        }
        a = fdsm(davis_southern_women_graph, **kw)
        b = fdsm(davis_southern_women_graph, **kw)
        assert all("sign" in d for _, _, d in a.edges(data=True))
        assert {(u, v): d["sign"] for u, v, d in a.edges(data=True)} == {
            (u, v): d["sign"] for u, v, d in b.edges(data=True)
        }


class TestSDSMEdgeConstraints:
    def test_constraints_change_null(
        self, davis_southern_women_graph, davis_women_nodes
    ):
        B = davis_southern_women_graph
        women = davis_women_nodes
        artifacts = sorted(
            (n for n, d in B.nodes(data=True) if d["bipartite"] == 1), key=str
        )
        present = next((women[0], f) for f in artifacts if B.has_edge(women[0], f))
        absent = next((women[0], f) for f in artifacts if not B.has_edge(women[0], f))

        base = sdsm(B, agent_nodes=women)
        prohibited = sdsm(B, agent_nodes=women, prohibited=[absent])
        required = sdsm(B, agent_nodes=women, required=[present])

        diff_pro = sum(
            1
            for u, v, d in prohibited.edges(data=True)
            if abs(d["sdsm_pvalue"] - base[u][v]["sdsm_pvalue"]) > 1e-9
        )
        diff_req = sum(
            1
            for u, v, d in required.edges(data=True)
            if abs(d["sdsm_pvalue"] - base[u][v]["sdsm_pvalue"]) > 1e-9
        )
        assert diff_pro > 0 and diff_req > 0

    def test_constraints_compose_with_signed(
        self, davis_southern_women_graph, davis_women_nodes
    ):
        artifact = next(n for n in davis_southern_women_graph[davis_women_nodes[0]])
        H = sdsm(
            davis_southern_women_graph,
            agent_nodes=davis_women_nodes,
            signed=True,
            required=[(davis_women_nodes[0], artifact)],
        )
        assert all("sign" in d for _, _, d in H.edges(data=True))
