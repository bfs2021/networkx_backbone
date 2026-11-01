"""Regression tests for the bipartite null models (SDSM/BiCM, FDSM, fixed*)."""

import itertools
import warnings

import networkx as nx
import numpy as np
import pytest
from scipy import stats

import networkx_backbone as nb
from networkx_backbone import bipartite as bp


def _davis_matrix(B, women):
    agents, artifacts, R, observed = bp._bipartite_projection_matrix(B, women)
    return agents, artifacts, R, observed


def _brute_pb_pmf(p):
    """Poisson-binomial pmf by enumerating all outcomes."""
    pmf = np.zeros(len(p) + 1)
    for bits in itertools.product([0, 1], repeat=len(p)):
        pmf[sum(bits)] += np.prod([q if b else 1 - q for q, b in zip(p, bits)])
    return pmf


# ---------------------------------------------------------------------------
# BiCM
# ---------------------------------------------------------------------------


class TestBiCM:
    def test_margins_match_observed_degrees(self, davis_southern_women_graph, davis_women_nodes):
        B = davis_southern_women_graph
        P, agents, artifacts = nb.bicm(B, davis_women_nodes, return_labels=True)
        row = np.array([B.degree(a) for a in agents])
        col = np.array([B.degree(f) for f in artifacts])
        np.testing.assert_allclose(P.sum(axis=1), row, atol=1e-6)
        np.testing.assert_allclose(P.sum(axis=0), col, atol=1e-6)
        assert np.all((P > 0) & (P < 1))

    def test_is_maximum_entropy_form(self, davis_southern_women_graph, davis_women_nodes):
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

    def test_converged_fit_does_not_warn(self, davis_southern_women_graph, davis_women_nodes):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            nb.bicm(davis_southern_women_graph, davis_women_nodes)


# ---------------------------------------------------------------------------
# Poisson-binomial helpers and SDSM p-values
# ---------------------------------------------------------------------------


class TestPoissonBinomial:
    @pytest.mark.parametrize("seed", range(4))
    def test_exact_matches_brute_force(self, seed):
        p = np.random.default_rng(seed).random(9)
        pmf = _brute_pb_pmf(p)
        np.testing.assert_allclose(bp._poisson_binomial_pmf(p), pmf, atol=1e-13)
        for k in range(-1, 11):
            expected_sf = pmf[max(k, 0):].sum()
            expected_cdf = pmf[: max(k + 1, 0)].sum()
            assert bp._poisson_binomial_sf(p, k) == pytest.approx(expected_sf, abs=1e-12)
            assert bp._poisson_binomial_cdf(p, k) == pytest.approx(expected_cdf, abs=1e-12)

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
            [("a", 1), ("a", 2), ("a", 3), ("b", 1), ("b", 2), ("c", 3), ("c", 4),
             ("d", 1), ("d", 4), ("d", 5), ("e", 2), ("e", 5)]
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
                    hi, lo = pmf[obs[i, j]:].sum(), pmf[: obs[i, j] + 1].sum()
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
            (agents[i], artifacts[k]) for i in range(1, 4) for k in range(5) if R[i, k] == 0
        ]
        return agents, artifacts, R, required, prohibited

    def test_refit_preserves_margins(self, davis_southern_women_graph, davis_women_nodes):
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

    def test_constraints_accept_numpy_array(self, davis_southern_women_graph, davis_women_nodes):
        _, _, _, _, prohibited = self._cells(davis_southern_women_graph, davis_women_nodes)
        as_list = nb.sdsm(davis_southern_women_graph, davis_women_nodes, prohibited=prohibited)
        as_array = nb.sdsm(
            davis_southern_women_graph, davis_women_nodes, prohibited=np.array(prohibited)
        )
        for u, v, p in as_list.edges(data="sdsm_pvalue"):
            assert as_array[u][v]["sdsm_pvalue"] == pytest.approx(p)

    def test_reversed_pair_raises(self, davis_southern_women_graph, davis_women_nodes):
        _, _, _, _, prohibited = self._cells(davis_southern_women_graph, davis_women_nodes)
        a, f = prohibited[0]
        with pytest.raises(nx.NetworkXError, match="ordered"):
            nb.sdsm(davis_southern_women_graph, davis_women_nodes, prohibited=[(f, a)])

    @pytest.mark.parametrize("bad", [["abc"], [("x",)], [("a", "b", "c")], [5]])
    def test_malformed_pairs_raise(self, davis_southern_women_graph, davis_women_nodes, bad):
        with pytest.raises(nx.NetworkXError):
            nb.sdsm(davis_southern_women_graph, davis_women_nodes, required=bad)

    def test_agent_agent_pair_raises(self, davis_southern_women_graph, davis_women_nodes):
        pair = (davis_women_nodes[0], davis_women_nodes[1])
        with pytest.raises(nx.NetworkXError):
            nb.sdsm(davis_southern_women_graph, davis_women_nodes, prohibited=[pair])

    def test_contradictory_constraints_raise(self, davis_southern_women_graph, davis_women_nodes):
        _, _, _, required, prohibited = self._cells(davis_southern_women_graph, davis_women_nodes)
        with pytest.raises(nx.NetworkXError):
            nb.sdsm(davis_southern_women_graph, davis_women_nodes, prohibited=required[:1])
        with pytest.raises(nx.NetworkXError):
            nb.sdsm(davis_southern_women_graph, davis_women_nodes, required=prohibited[:1])

    def test_empty_constraints_equal_unconstrained(
        self, davis_southern_women_graph, davis_women_nodes
    ):
        base = nb.sdsm(davis_southern_women_graph, davis_women_nodes)
        empty = nb.sdsm(
            davis_southern_women_graph, davis_women_nodes, prohibited=[], required=np.empty((0, 2))
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
            davis_southern_women_graph, davis_women_nodes, trials=trials, seed=0, signed=signed
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

    @pytest.mark.parametrize(
        "make_seed",
        [lambda: 3, lambda: np.random.default_rng(3), lambda: np.random.RandomState(3)],
    )
    def test_seed_types_reproducible(self, davis_southern_women_graph, davis_women_nodes, make_seed):
        a = nb.fdsm(davis_southern_women_graph, davis_women_nodes, trials=30, seed=make_seed())
        b = nb.fdsm(davis_southern_women_graph, davis_women_nodes, trials=30, seed=make_seed())
        assert dict(nx.get_edge_attributes(a, "fdsm_pvalue")) == dict(
            nx.get_edge_attributes(b, "fdsm_pvalue")
        )

    def test_negative_burn_in_raises(self, bipartite_graph):
        with pytest.raises(ValueError):
            nb.fdsm(bipartite_graph, [1, 2, 3], trials=5, burn_in=-1)


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
    @pytest.mark.parametrize("dims", [(3, 3, 4), (4, 3, 7), (3, 4, 5), (2, 4, 6), (4, 4, 3)])
    def test_fixedfill_pmf_matches_enumeration(self, dims):
        na, nf, T = dims
        counts = np.zeros(nf + 1)
        for cells in itertools.combinations(range(na * nf), T):
            M = np.zeros(na * nf, dtype=int)
            M[list(cells)] = 1
            M = M.reshape(na, nf)
            counts[M[0] @ M[1]] += 1
        np.testing.assert_allclose(bp._fixedfill_pmf(na, nf, T), counts / counts.sum(), atol=1e-12)

    def test_fixedfill_pvalues(self):
        B, agents, _, R = _random_bipartite(3, 4, 0.5, seed=2)
        pmf = bp._fixedfill_pmf(3, 4, int(R.sum()))
        H = nb.fixedfill(B, agents, alpha=np.inf, signed=False)
        obs = R @ R.T
        for i, j in itertools.combinations(range(3), 2):
            assert H[agents[i]][agents[j]]["fixedfill_pvalue"] == pytest.approx(
                pmf[obs[i, j]:].sum()
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
# Input validation
# ---------------------------------------------------------------------------


_GRAPH_FUNCS = [
    nb.simple_projection,
    nb.hyper_projection,
    nb.probs_projection,
    nb.ycn_projection,
    nb.bipartite_projection,
    nb.bicm,
    nb.sdsm,
    lambda B, a: nb.fdsm(B, a, trials=5, seed=0),
    nb.fixedfill,
    nb.fixedrow,
    nb.fixedcol,
    lambda B, a: nb.backbone_from_projection(B, a, method="fixedrow"),
]


class TestValidation:
    @pytest.mark.parametrize("func", _GRAPH_FUNCS)
    def test_generator_agent_nodes(self, func, davis_southern_women_graph, davis_women_nodes):
        from_list = func(davis_southern_women_graph, list(davis_women_nodes))
        from_gen = func(davis_southern_women_graph, (w for w in davis_women_nodes))
        if isinstance(from_list, np.ndarray):
            np.testing.assert_allclose(from_list, from_gen)
        else:
            assert set(from_gen.nodes()) == set(davis_women_nodes)
            assert {frozenset(e) for e in from_gen.edges()} == {
                frozenset(e) for e in from_list.edges()
            }

    @pytest.mark.parametrize("func", _GRAPH_FUNCS)
    @pytest.mark.parametrize("agents", [["a", "x"], ["a", "b"]])
    def test_agent_set_not_a_side_raises(self, func, agents):
        B = nx.Graph([("a", "x"), ("b", "x"), ("a", "y"), ("c", "y")])
        with pytest.raises(nx.NetworkXError, match="one side"):
            func(B, agents)

    @pytest.mark.parametrize("func", _GRAPH_FUNCS)
    @pytest.mark.parametrize("cls", [nx.DiGraph, nx.MultiGraph])
    def test_directed_and_multigraph_rejected(self, func, cls):
        B = cls([("a", "x"), ("b", "x"), ("a", "y")])
        with pytest.raises(nx.NetworkXNotImplemented):
            func(B, ["a", "b"])

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
    @pytest.mark.parametrize("method", ["sdsm", "fdsm", "fixedfill", "fixedrow", "fixedcol"])
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
            davis_southern_women_graph, davis_women_nodes, alpha=np.inf, signed=signed, **kw
        )
        expected = {frozenset((u, v)) for u, v, p in scored.edges(data=attr) if p < alpha}
        assert {frozenset(e) for e in H.edges()} == expected
        assert H.number_of_edges() < len(davis_women_nodes) * (len(davis_women_nodes) - 1) // 2

    @pytest.mark.parametrize("method", ["sdsm", "fixedrow"])
    def test_mtc_forwarded(self, davis_southern_women_graph, davis_women_nodes, method):
        alpha = 0.3
        scored = getattr(nb, method)(davis_southern_women_graph, davis_women_nodes, alpha=np.inf)
        attr = f"{method}_pvalue"
        edges = list(scored.edges(data=attr))
        adjusted = nb.adjust_pvalues([p for *_, p in edges], "bonferroni")
        expected = {frozenset((u, v)) for (u, v, _), q in zip(edges, adjusted) if q < alpha}
        H = nb.backbone_from_projection(
            davis_southern_women_graph, davis_women_nodes, method=method, alpha=alpha,
            mtc="bonferroni",
        )
        assert {frozenset(e) for e in H.edges()} == expected

    def test_backbone_auto_dispatch_thresholds(self, davis_southern_women_graph):
        H = nb.backbone(davis_southern_women_graph, alpha=0.1)
        assert all(d["sdsm_pvalue"] < 0.1 for *_, d in H.edges(data=True))
        assert H.number_of_edges() < 153
