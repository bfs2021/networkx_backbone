"""Tests for hypergraph backbone extraction (MDL method, Kirkley et al. 2026)."""

import itertools
import math
import random

import pytest

from networkx_backbone import (
    HypergraphBackbone,
    ValidatedHypergraph,
    hypergraph_compression_ratio,
    intersection_graph,
    maximal_hyperedges,
    mdl_hypergraph_backbone,
    order_filter,
    s_components,
    statistically_validated_cores,
    statistically_validated_hypergraph,
    svc,
    svh,
)
from networkx_backbone import hypergraph as hg
from networkx_backbone.hypergraph import (
    _bh_threshold,
    _child_codelength,
    _log2_binom,
    _parent_codelength,
    _reduced_mutual_information,
    _weight_codelength,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def simplex_with_subfaces(nodes, min_order=4):
    """A top face plus all of its nested sub-hyperedges down to ``min_order``."""
    nodes = list(nodes)
    edges = [tuple(nodes)]
    for k in range(min_order, len(nodes)):
        edges.extend(itertools.combinations(nodes, k))
    return edges


# ---------------------------------------------------------------------------
# Information-theoretic primitives
# ---------------------------------------------------------------------------


def test_rmi_symmetric_and_identity():
    n_nodes = 12
    for size_p, size_c, overlap in [
        (5, 4, 4),
        (5, 3, 3),
        (5, 2, 2),
        (4, 4, 3),
        (6, 3, 2),
    ]:
        case = (size_p, size_c, overlap)
        r_pc = _reduced_mutual_information(size_p, size_c, overlap, n_nodes)
        r_cp = _reduced_mutual_information(size_c, size_p, overlap, n_nodes)
        assert r_pc == pytest.approx(r_cp), case  # R is symmetric (Eqs. 9-10)

        # R(c, p) == H(c) - H(c|p)  (Eq. 9), using L=1 so the log L term cancels.
        h_c = _parent_codelength(size_c, 1, n_nodes)
        h_c_given_p = _child_codelength(size_c, size_p, overlap, 1, n_nodes)
        assert r_pc == pytest.approx(h_c - h_c_given_p), case


def test_rmi_positive_for_nesting_when_universe_large():
    # Nested c ⊂ p gives positive RMI once N is sufficiently larger than |p|.
    assert _reduced_mutual_information(5, 4, 4, 20) > 0
    # ...and can be non-positive when the universe is tiny.
    assert _reduced_mutual_information(5, 4, 4, 5) <= 0


# ---------------------------------------------------------------------------
# intersection_graph
# ---------------------------------------------------------------------------


def test_intersection_graph_basic():
    I = intersection_graph([(1, 2, 3), (2, 3, 4), (5, 6)])
    assert I.number_of_nodes() == 3
    assert I.number_of_edges() == 1  # only the first two overlap
    assert I[0][1]["overlap"] == 2
    assert I.nodes[0]["members"] == frozenset({1, 2, 3})


def test_intersection_graph_disjoint_has_no_edges():
    I = intersection_graph([(0, 1), (2, 3), (4, 5)])
    assert I.number_of_nodes() == 3
    assert I.number_of_edges() == 0


def test_intersection_graph_s_threshold():
    G = [(1, 2, 3), (2, 3, 4), (5, 6)]
    assert intersection_graph(G, s=1).number_of_edges() == 1
    assert intersection_graph(G, s=2).number_of_edges() == 1  # overlap is exactly 2
    assert intersection_graph(G, s=3).number_of_edges() == 0


def test_intersection_graph_invalid_s_raises():
    with pytest.raises(ValueError):
        intersection_graph([(1, 2, 3)], s=0)


# ---------------------------------------------------------------------------
# Structural methods: inclusion reduction, order filter, s-components
# ---------------------------------------------------------------------------


def test_maximal_hyperedges_removes_subsets():
    result = maximal_hyperedges([(1, 2, 3), (1, 2), (2, 3), (4, 5), (4, 5, 6)])
    assert set(result) == {frozenset({1, 2, 3}), frozenset({4, 5, 6})}
    # Ordered by decreasing size.
    assert [len(e) for e in result] == sorted((len(e) for e in result), reverse=True)


def test_maximal_hyperedges_nested_chain():
    assert maximal_hyperedges([(1,), (1, 2), (1, 2, 3)]) == [frozenset({1, 2, 3})]


def test_maximal_hyperedges_all_maximal():
    result = maximal_hyperedges([(1, 2), (3, 4), (5, 6)])
    assert set(result) == {frozenset({1, 2}), frozenset({3, 4}), frozenset({5, 6})}


def test_maximal_hyperedges_merges_duplicates():
    assert maximal_hyperedges([(1, 2, 3), (3, 2, 1)]) == [frozenset({1, 2, 3})]


def test_order_filter_min_max_and_orders():
    G = [(1, 2), (1, 2, 3), (1, 2, 3, 4)]
    assert set(order_filter(G, min_order=3)) == {
        frozenset({1, 2, 3}),
        frozenset({1, 2, 3, 4}),
    }
    assert order_filter(G, max_order=2) == [frozenset({1, 2})]
    assert order_filter(G, min_order=3, max_order=3) == [frozenset({1, 2, 3})]
    assert set(order_filter(G, orders=[2, 4])) == {
        frozenset({1, 2}),
        frozenset({1, 2, 3, 4}),
    }


def test_order_filter_invalid_range_raises():
    with pytest.raises(ValueError):
        order_filter([(1, 2, 3)], min_order=4, max_order=2)


def test_s_components_threshold():
    G = [(1, 2, 3), (2, 3, 4), (5, 6, 7)]
    # s=1 and s=2: first two hyperedges connect; the third is isolated.
    for s in (1, 2):
        comps = s_components(G, s=s)
        assert [len(c) for c in comps] == [2, 1]
    # s=3: no pair shares 3 nodes, so every hyperedge is its own component.
    assert [len(c) for c in s_components(G, s=3)] == [1, 1, 1]


def test_s_components_chain():
    comps = s_components([(1, 2), (2, 3), (3, 4), (10, 11)], s=1)
    assert [len(c) for c in comps] == [3, 1]
    # The big component holds the chain; the disjoint pair stands alone.
    assert frozenset({10, 11}) in comps[1]


# ---------------------------------------------------------------------------
# Core MDL backbone behaviour
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["edge", "node", "auto"])
def test_recovers_top_faces_of_nested_simplices(method):
    # Two disjoint 5-simplices, each with all size-4 sub-faces. The MDL backbone
    # should keep exactly the two top faces and prune every redundant sub-face.
    a, b = range(5), range(5, 10)
    G = simplex_with_subfaces(a) + simplex_with_subfaces(b)

    result = mdl_hypergraph_backbone(G, method=method)

    assert isinstance(result, HypergraphBackbone)
    assert len(result.backbone) == 2
    assert frozenset(a) in result.backbone
    assert frozenset(b) in result.backbone
    assert result.compression_ratio < 1.0
    assert result.description_length <= result.baseline_description_length
    assert result.n_nodes == 10
    assert result.n_input_hyperedges == len(G)


def test_downward_closure_compresses_and_keeps_tops():
    a, b = range(5), range(5, 10)
    G = simplex_with_subfaces(a, min_order=2) + simplex_with_subfaces(b, min_order=2)

    result = mdl_hypergraph_backbone(G)

    assert frozenset(a) in result.backbone
    assert frozenset(b) in result.backbone
    assert result.fraction_kept < 0.5  # substantial sparsification
    assert 0.0 <= result.compression_ratio < 1.0


def test_disjoint_hyperedges_are_incompressible():
    G = [(0, 1), (2, 3), (4, 5)]
    result = mdl_hypergraph_backbone(G)
    assert len(result.backbone) == 3
    assert result.assignment == {}
    assert result.compression_ratio == pytest.approx(1.0)


def test_description_length_never_exceeds_baseline():
    G = simplex_with_subfaces(range(6), min_order=2)
    result = mdl_hypergraph_backbone(G)
    assert result.description_length <= result.baseline_description_length
    assert 0.0 <= result.compression_ratio <= 1.0


def test_compression_ratio_helper_matches_backbone():
    G = simplex_with_subfaces(range(5)) + simplex_with_subfaces(range(5, 10))
    eta = hypergraph_compression_ratio(G)
    assert eta == pytest.approx(mdl_hypergraph_backbone(G).compression_ratio)
    assert 0.0 <= eta <= 1.0


def test_deterministic():
    G = simplex_with_subfaces(range(5), min_order=2) + simplex_with_subfaces(
        range(5, 10), min_order=2
    )
    r1 = mdl_hypergraph_backbone(G)
    r2 = mdl_hypergraph_backbone(G)
    assert set(r1.backbone) == set(r2.backbone)


# ---------------------------------------------------------------------------
# Weighted model
# ---------------------------------------------------------------------------


def _family_a_with_filler():
    """Family A (top + one size-4 child) plus a disjoint filler simplex.

    The filler raises the node count so the size-4 child of A is topologically
    redundant (pruned) in the unweighted backbone.
    """
    A = [(0, 1, 2, 3, 4), (0, 1, 2, 3)]
    filler = simplex_with_subfaces(range(5, 10))
    return A, filler


def test_gamma_one_equivalent_to_unweighted():
    A, filler = _family_a_with_filler()
    G = A + filler
    weights = [5.0, 5.0] + [5.0] * len(filler)  # uniform weights

    unweighted = mdl_hypergraph_backbone(G)
    g1 = mdl_hypergraph_backbone(G, weights=weights, gamma=1.0)

    assert g1.weighted is True
    assert set(g1.backbone) == set(unweighted.backbone)


def test_low_gamma_forces_high_weight_hyperedge_into_backbone():
    A, filler = _family_a_with_filler()
    G = A + filler
    weights = [1.0, 100.0] + [1.0] * len(filler)  # the size-4 child is heavy

    unweighted = mdl_hypergraph_backbone(G)
    weighted = mdl_hypergraph_backbone(G, weights=weights, gamma=0.01)

    child = frozenset({0, 1, 2, 3})
    assert child not in unweighted.backbone  # topologically redundant
    assert child in weighted.backbone  # weight keeps it in the backbone


def test_geometric_prior_runs():
    A, filler = _family_a_with_filler()
    G = A + filler
    weights = [1.0, 100.0] + [1.0] * len(filler)
    result = mdl_hypergraph_backbone(G, weights=weights, gamma=0.05, prior="geometric")
    assert frozenset({0, 1, 2, 3}) in result.backbone


# ---------------------------------------------------------------------------
# Edge cases and input handling
# ---------------------------------------------------------------------------


def test_empty_hypergraph():
    result = mdl_hypergraph_backbone([])
    assert len(result.backbone) == 0
    assert result.compression_ratio == pytest.approx(1.0)


def test_single_hyperedge():
    result = mdl_hypergraph_backbone([(1, 2, 3)])
    assert result.backbone == [frozenset({1, 2, 3})]
    assert result.assignment == {}
    assert result.compression_ratio == pytest.approx(1.0)


def test_duplicate_hyperedges_are_merged():
    result = mdl_hypergraph_backbone([(1, 2, 3), (1, 2, 3), (3, 2, 1)])
    assert result.backbone == [frozenset({1, 2, 3})]
    assert result.n_input_hyperedges == 1


def test_repeated_nodes_within_hyperedge_ignored():
    result = mdl_hypergraph_backbone([(1, 1, 2, 2, 3)])
    assert result.backbone == [frozenset({1, 2, 3})]


def test_duplicate_weighted_hyperedges_sum_weights():
    # Two copies of the same hyperedge merge; their weights add (3 + 4 = 7 > 1),
    # so the merged hyperedge is treated as weighted.
    result = mdl_hypergraph_backbone(
        [(1, 2, 3), (1, 2, 3)], weights=[3.0, 4.0], gamma=0.5
    )
    assert result.backbone == [frozenset({1, 2, 3})]
    assert result.weighted is True


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_invalid_gamma_raises():
    for gamma in (0.0, 1.5):  # either side of (0, 1]
        with pytest.raises(ValueError, match="gamma"):
            mdl_hypergraph_backbone([(1, 2, 3)], weights=[2.0], gamma=gamma)


def test_invalid_prior_raises():
    with pytest.raises(ValueError):
        mdl_hypergraph_backbone([(1, 2, 3)], weights=[2.0], prior="bogus")


def test_invalid_method_raises():
    with pytest.raises(ValueError):
        mdl_hypergraph_backbone([(1, 2, 3)], method="bogus")


def test_mdl_auto_is_best_of_edge_and_node():
    # On a downward-closed simplex the two greedy schemes differ; "auto" must
    # match the lower-description-length one.
    G = simplex_with_subfaces(range(6), min_order=2)
    edge = mdl_hypergraph_backbone(G, method="edge").description_length
    node = mdl_hypergraph_backbone(G, method="node").description_length
    auto = mdl_hypergraph_backbone(G, method="auto").description_length
    assert auto == pytest.approx(min(edge, node))


def test_mdl_node_weighting_keeps_high_weight_child():
    # The gamma weight knob must work under the node optimiser too.
    A = [(0, 1, 2, 3, 4), (0, 1, 2, 3)]
    filler = simplex_with_subfaces(range(5, 10))
    G = A + filler
    weights = [1.0, 100.0] + [1.0] * len(filler)
    result = mdl_hypergraph_backbone(G, weights=weights, gamma=0.01, method="node")
    assert frozenset({0, 1, 2, 3}) in result.backbone


def test_weights_length_mismatch_raises():
    with pytest.raises(ValueError):
        mdl_hypergraph_backbone([(1, 2, 3), (2, 3, 4)], weights=[1.0])


def test_weight_below_one_raises():
    with pytest.raises(ValueError):
        mdl_hypergraph_backbone([(1, 2, 3)], weights=[0.5])


# ---------------------------------------------------------------------------
# Statistically validated hypergraphs (SVH) and cores (SVC)
# ---------------------------------------------------------------------------


def test_svh_pvalue_matches_binomial():
    from scipy.stats import binom

    edges = [(1, 2)] * 5 + [(1, 3), (2, 4), (3, 4), (5, 6)]
    result = statistically_validated_hypergraph(edges)
    # order-2 instances N=9; node 1 and 2 each have degree 6; {1,2} co-occurs 5x.
    expected = binom.sf(5 - 1, 9, (6 / 9) * (6 / 9))
    assert result.pvalues[frozenset({1, 2})] == pytest.approx(expected)


def test_svh_validates_over_represented_pair():
    # {1,2} recurs 5x; a high-multiplicity background pair inflates the instance
    # count so the planted pair is significant while the background is not.
    edges = [(1, 2)] * 5 + [(3, 4)] * 100
    result = statistically_validated_hypergraph(edges, alpha=0.05)
    assert isinstance(result, ValidatedHypergraph)
    assert frozenset({1, 2}) in result.validated
    assert frozenset({3, 4}) not in result.validated  # expected background
    assert result.method == "svh"


def test_svh_weights_equal_repeated_edges():
    repeated = statistically_validated_hypergraph([(1, 2)] * 5 + [(3, 4)] * 100)
    weighted = statistically_validated_hypergraph([(1, 2), (3, 4)], weights=[5, 100])
    assert weighted.pvalues[frozenset({1, 2})] == pytest.approx(
        repeated.pvalues[frozenset({1, 2})]
    )
    assert set(weighted.validated) == set(repeated.validated)


def test_svh_alpha_monotonic():
    edges = [(1, 2)] * 5 + [(3, 4)] * 100
    strict = set(svh(edges, alpha=0.001).validated)
    loose = set(svh(edges, alpha=0.5).validated)
    assert strict <= loose


def test_svh_single_occurrences_not_validated():
    result = statistically_validated_hypergraph([(1, 2), (3, 4), (5, 6)], alpha=0.05)
    assert result.validated == []


def test_svh_max_order_restricts_tests():
    edges = [(1, 2)] * 5 + [(3, 4)] * 100 + [(5, 6, 7)] * 5
    result = statistically_validated_hypergraph(edges, max_order=2)
    assert all(len(e) == 2 for e in result.pvalues)


def test_svh_empty():
    result = statistically_validated_hypergraph([])
    assert len(result) == 0
    assert list(result) == []


def test_svc_validates_core_and_drops_subgroups():
    edges = [(1, 2, 3)] * 5 + [(7, 8)] * 200
    result = statistically_validated_cores(edges, alpha=0.05)
    assert result.method == "svc"
    assert frozenset({1, 2, 3}) in result.validated
    # Sub-pairs of a validated core are not separately validated.
    assert frozenset({1, 2}) not in result.validated
    assert frozenset({2, 3}) not in result.validated


def test_svc_counts_subgroup_cooccurrence():
    # {1,2} appears inside both a triangle and a separate pair: co-occurrence 2.
    edges = [(1, 2, 3), (1, 2)]
    result = statistically_validated_cores(edges, min_order=2)
    assert result.counts[frozenset({1, 2})] == 2


@pytest.mark.parametrize("func", [svh, svc])
def test_svh_svc_invalid_multiplicities(func):
    with pytest.raises(ValueError, match="same length"):
        func([(1, 2), (3, 4)], weights=[1])
    for bad in (1.5, 0, 0.5, float("nan")):  # non-integer, or below one
        with pytest.raises(ValueError, match="integer"):
            func([(1, 2), (3, 4)], weights=[2, bad])


@pytest.mark.parametrize("func", [svh, svc])
def test_svh_svc_alpha_validated(func):
    for alpha in (0.0, 1.5):  # either side of (0, 1]
        with pytest.raises(ValueError, match="alpha"):
            func([(1, 2)] * 3, alpha=alpha)
    assert func([(1, 2)] * 3, alpha=1.0).alpha == 1.0


def test_svh_svc_aliases():
    edges = [(1, 2)] * 5 + [(3, 4)] * 100
    assert set(svh(edges).validated) == set(
        statistically_validated_hypergraph(edges).validated
    )
    assert set(svc(edges).validated) == set(
        statistically_validated_cores(edges).validated
    )


# ---------------------------------------------------------------------------
# Regression tests (review fixes)
# ---------------------------------------------------------------------------


def test_child_codelength_matches_documented_equation():
    # H(c|p) = log L + log min(|p|,|c|) + log C(|p|,|p∩c|) + log C(N-|p|,|c|-|p∩c|)
    L, N, sp, sc, o = 3, 20, 6, 4, 2
    expected = (
        math.log2(L)
        + math.log2(min(sp, sc))
        + math.log2(math.comb(sp, o))
        + math.log2(math.comb(N - sp, sc - o))
    )
    assert _child_codelength(sc, sp, o, L, N) == pytest.approx(expected)
    assert _parent_codelength(sc, L, N) == pytest.approx(
        math.log2(L) + math.log2(math.comb(N, sc))
    )


def test_log2_binom_out_of_range_is_minus_inf():
    assert _log2_binom(3, 5) == float("-inf")
    assert _log2_binom(3, -1) == float("-inf")
    assert _log2_binom(-1, 0) == float("-inf")


def test_weight_codelength_degenerate_means_and_bad_prior():
    # mean weight 1 -> expected weight mu == 1 -> zero-cost weights.
    assert _weight_codelength(1.0, 1, 0.5, 1.0, "poisson") == 0.0
    assert _weight_codelength(1.0, 0, 0.5, 1.0, "geometric") == 0.0
    with pytest.raises(ValueError, match="prior"):
        _weight_codelength(2.0, 1, 0.5, 2.0, "uniform")


def test_bh_threshold_empty():
    assert _bh_threshold([], 0.05, 10) == 0.0


def test_backbone_len_and_empty_hyperedges_skipped():
    result = mdl_hypergraph_backbone([(1, 2, 3), (), (1, 2)], weights=[1, 5, 1])
    assert len(result) == len(result.backbone)
    assert result.n_input_hyperedges == 2


MIXED = [(1, "a", 2), (1, "a"), ("b", 3), (("t", 1), 3)]


def test_mixed_type_labels_supported():
    # Regression: sorting mixed int/str/tuple labels raised TypeError.
    maximal = maximal_hyperedges(MIXED)
    assert set(maximal) == {
        frozenset({1, "a", 2}),
        frozenset({"b", 3}),
        frozenset({("t", 1), 3}),
    }
    assert len(maximal[0]) == 3
    comps = s_components([*MIXED, (4, "c")])
    assert [len(c) for c in comps] == [2, 2, 1]
    assert maximal_hyperedges(MIXED) == maximal_hyperedges(list(reversed(MIXED)))
    assert [set(c) for c in s_components(MIXED)] == [
        set(c) for c in s_components(list(reversed(MIXED)))
    ]
    result = mdl_hypergraph_backbone(MIXED)
    assert frozenset({1, "a", 2}) in result.backbone
    assert intersection_graph(MIXED).number_of_edges() == 2


def test_homogeneous_but_incomparable_hyperedges():
    # Each hyperedge is internally sortable, but ints vs strs across edges.
    edges = [(1, 2), ("a", "b"), (3, 4)]
    assert set(maximal_hyperedges(edges)) == {frozenset(e) for e in edges}
    assert len(s_components(edges)) == 3


def test_natural_order_kept_for_comparable_labels():
    # Integer ties are still ordered numerically (not by repr: "10" < "9").
    assert maximal_hyperedges([(10, 11), (9, 12)]) == [
        frozenset({9, 12}),
        frozenset({10, 11}),
    ]
    assert s_components([(10, 11), (9, 12)]) == [
        [frozenset({9, 12})],
        [frozenset({10, 11})],
    ]


def test_svh_svc_skip_empty_hyperedges():
    a = svh([(1, 2), (), (1, 2)], weights=[2, 7, 3])
    assert a.counts == {frozenset({1, 2}): 5}
    assert a.n_instances == 5
    assert len(svc([(), ()])) == 0
    assert svc([]).n_instances == 0


def test_svc_large_hyperedge_guard():
    # Regression: a single 24-node hyperedge (~16.7M sub-groups) never finished.
    big = [tuple(range(24))] + [(0, 1)] * 3
    with pytest.raises(ValueError, match="max_order"):
        svc(big)
    bounded = svc(big, max_order=3)
    assert max(len(g) for g in bounded.pvalues) == 3
    assert len(bounded.pvalues) == math.comb(24, 3) + math.comb(24, 2)
    with pytest.raises(ValueError, match="max_groups"):
        svc(big, max_order=3, max_groups=100)


def test_svc_max_groups_none_disables_guard():
    edges = [tuple(range(6))] * 3 + [(7, 8)]
    assert svc(edges, max_groups=None).pvalues == svc(edges).pvalues
    with pytest.raises(ValueError):
        svc(edges, max_groups=10)


def test_svc_min_order_validated():
    with pytest.raises(ValueError, match="min_order"):
        svc([(1, 2, 3)], min_order=0)
    assert all(len(g) >= 1 for g in svc([(1, 2, 3)], min_order=1).pvalues)


def test_svc_order_with_all_groups_dropped():
    # Once {1,2,3} is validated, every order-2 sub-group is dropped, so there is
    # nothing left to test at order 2.
    edges = [(1, 2, 3)] * 6 + [(i,) for i in range(10, 400)]
    result = svc(edges, alpha=0.05)
    assert result.validated == [frozenset({1, 2, 3})]
    assert all(len(g) == 3 for g in result.pvalues)


def test_svh_svc_vectorised_pvalues_match_scalar():
    from scipy.stats import binom

    rng = random.Random(3)
    edges = [tuple(rng.sample(range(15), rng.randint(2, 4))) for _ in range(60)]
    weights = [rng.randint(1, 5) for _ in edges]
    result = svh(edges, weights=weights)
    norm, mult = hg._normalize_multiplicities(edges, weights)
    for order in {len(e) for e in norm}:
        members = [(e, w) for e, w in zip(norm, mult) if len(e) == order]
        n = sum(w for _, w in members)
        degree = {}
        for e, w in members:
            for v in e:
                degree[v] = degree.get(v, 0) + w
        for e, w in members:
            scalar = hg._svh_pvalue(w, n, [degree[v] for v in e], binom)
            assert result.pvalues[e] == scalar


def _reference_greedy_node(n_edges, neighbors, wterm):
    """The original O(rounds * (m + P)) node greedy, kept as an oracle."""
    in_backbone = [False] * n_edges
    best_saving = [0.0] * n_edges
    best_parent = [None] * n_edges

    def _attach_children(parent):
        for c, rmi in neighbors[parent]:
            if in_backbone[c]:
                continue
            gain = rmi + wterm[c]
            if gain > best_saving[c]:
                best_saving[c] = gain
                best_parent[c] = parent

    while True:
        chosen, best_gain = None, 1e-12
        for e in range(n_edges):
            if in_backbone[e]:
                continue
            gain = -best_saving[e]
            for c, rmi in neighbors[e]:
                if in_backbone[c]:
                    continue
                delta = (rmi + wterm[c]) - best_saving[c]
                if delta > 0.0:
                    gain += delta
            if gain > best_gain:
                best_gain, chosen = gain, e
        if chosen is None:
            break
        in_backbone[chosen] = True
        best_saving[chosen] = 0.0
        best_parent[chosen] = None
        _attach_children(chosen)

    for e in range(n_edges):
        if not in_backbone[e] and best_parent[e] is None:
            in_backbone[e] = True
            _attach_children(e)
    return {e: best_parent[e] for e in range(n_edges) if not in_backbone[e]}


def _random_hypergraph(seed):
    rng = random.Random(seed)
    n_nodes = rng.randint(4, 30)
    m = rng.randint(2, 80)
    edges = [
        tuple(rng.sample(range(n_nodes), rng.randint(1, min(n_nodes, 6))))
        for _ in range(m)
    ]
    weights = [rng.choice([1, 1, 2, 3, 10]) for _ in edges] if seed % 2 else None
    gamma = rng.choice([1.0, 0.5, 0.1])
    return edges, weights, gamma


@pytest.mark.parametrize("method", ["node", "auto"])
def test_fast_greedy_node_matches_reference(monkeypatch, method):
    for seed in range(120):
        edges, weights, gamma = _random_hypergraph(seed)
        fast = mdl_hypergraph_backbone(
            edges, weights=weights, gamma=gamma, method=method
        )
        with monkeypatch.context() as mp:
            mp.setattr(hg, "_greedy_node", _reference_greedy_node)
            ref = mdl_hypergraph_backbone(
                edges, weights=weights, gamma=gamma, method=method
            )
        assert fast.backbone == ref.backbone, seed
        assert fast.assignment == ref.assignment, seed
        assert fast.description_length == ref.description_length, seed


def test_greedy_node_scales():
    # The previous implementation took ~8 s here; the lazy heap takes well under 1 s.
    import time

    rng = random.Random(0)
    edges = [tuple(rng.sample(range(300), rng.randint(2, 6))) for _ in range(1500)]
    start = time.perf_counter()
    mdl_hypergraph_backbone(edges, method="node")
    assert time.perf_counter() - start < 5.0
