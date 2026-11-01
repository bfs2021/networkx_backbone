"""Les Miserables benchmark tests across non-bipartite backbone methods."""

import warnings

import networkx as nx
import pytest

import networkx_backbone as nb

DEFAULT_PVALUE = 0.05


def _warn_if_no_edge_reduction(method_name, original_edges, filtered_edges):
    if filtered_edges == original_edges:
        warnings.warn(
            (
                f"{method_name}: filtered graph has the same number of edges "
                f"as the original ({original_edges}). Re-test and validate."
            ),
            UserWarning,
            stacklevel=2,
        )


WEIGHTED_METHOD_EDGE_COUNTS = [
    (
        "disparity_filter",
        lambda G: nb.disparity_filter(G),
        lambda H: nb.threshold_filter(
            H, "disparity_pvalue", DEFAULT_PVALUE, mode="below"
        ),
        9,
    ),
    (
        "noise_corrected_filter",
        lambda G: nb.noise_corrected_filter(G),
        lambda H: nb.threshold_filter(H, "nc_score", 2.0, mode="above"),
        98,
    ),
    (
        "marginal_likelihood_filter",
        lambda G: nb.marginal_likelihood_filter(G),
        lambda H: nb.threshold_filter(H, "ml_pvalue", DEFAULT_PVALUE, mode="below"),
        126,
    ),
    (
        "ecm_filter",
        lambda G: nb.ecm_filter(G),
        lambda H: nb.threshold_filter(H, "ecm_pvalue", DEFAULT_PVALUE, mode="below"),
        72,
    ),
    (
        "lans_filter",
        lambda G: nb.lans_filter(G),
        lambda H: nb.threshold_filter(H, "lans_pvalue", DEFAULT_PVALUE, mode="below"),
        109,
    ),
    (
        "multiple_linkage_analysis(alpha=0.05)",
        lambda G: nb.multiple_linkage_analysis(G, alpha=0.05),
        lambda H: nb.boolean_filter(H, "mla_keep"),
        109,
    ),
    (
        "global_threshold_filter(threshold=2)",
        lambda G: nb.global_threshold_filter(G, threshold=2),
        lambda H: nb.boolean_filter(H, "global_threshold_keep"),
        157,
    ),
    (
        "strongest_n_ties(n=2)",
        lambda G: nb.strongest_n_ties(G, n=2),
        lambda H: nb.boolean_filter(H, "strongest_n_ties_keep"),
        113,
    ),
    (
        "global_sparsification(s=0.5)",
        lambda G: nb.global_sparsification(G, s=0.5),
        lambda H: nb.boolean_filter(H, "global_sparsification_keep"),
        127,
    ),
    (
        "primary_linkage_analysis",
        lambda G: nb.primary_linkage_analysis(G),
        lambda H: nb.boolean_filter(H, "primary_linkage_keep"),
        69,
    ),
    (
        "edge_betweenness_filter(s=0.5)",
        lambda G: nb.edge_betweenness_filter(G, s=0.5),
        lambda H: nb.boolean_filter(H, "edge_betweenness_keep"),
        127,
    ),
    (
        "node_degree_filter(min_degree=2)",
        lambda G: nb.node_degree_filter(G, min_degree=2),
        lambda H: nb.boolean_filter(H, "node_degree_keep"),
        237,
    ),
    (
        "high_salience_skeleton",
        lambda G: nb.high_salience_skeleton(G),
        lambda H: nb.threshold_filter(H, "salience", 0.5, mode="above"),
        76,
    ),
    (
        "metric_backbone",
        lambda G: nb.metric_backbone(G),
        lambda H: nb.boolean_filter(H, "metric_keep"),
        163,
    ),
    (
        "ultrametric_backbone",
        lambda G: nb.ultrametric_backbone(G),
        lambda H: nb.boolean_filter(H, "ultrametric_keep"),
        118,
    ),
    (
        "doubly_stochastic_filter",
        lambda G: nb.doubly_stochastic_filter(G),
        lambda H: nb.threshold_filter(H, "ds_weight", 0.1, mode="above"),
        109,
    ),
    (
        "h_backbone",
        lambda G: nb.h_backbone(G),
        lambda H: nb.boolean_filter(H, "h_backbone_keep"),
        18,
    ),
    (
        "modularity_backbone",
        lambda G: nb.modularity_backbone(G, seed=42),
        lambda H: nb.boolean_filter(H, "modularity_keep"),
        72,
    ),
    (
        "planar_maximally_filtered_graph",
        lambda G: nb.planar_maximally_filtered_graph(G),
        lambda H: nb.boolean_filter(H, "pmfg_keep"),
        162,
    ),
    (
        "maximum_spanning_tree_backbone",
        lambda G: nb.maximum_spanning_tree_backbone(G),
        lambda H: nb.boolean_filter(H, "mst_keep"),
        76,
    ),
    (
        "neighborhood_overlap",
        lambda G: nb.neighborhood_overlap(G),
        lambda H: nb.fraction_filter(H, "overlap", 0.3, ascending=False),
        76,
    ),
    (
        "jaccard_backbone",
        lambda G: nb.jaccard_backbone(G),
        lambda H: nb.fraction_filter(H, "jaccard", 0.3, ascending=False),
        76,
    ),
    (
        "dice_backbone",
        lambda G: nb.dice_backbone(G),
        lambda H: nb.fraction_filter(H, "dice", 0.3, ascending=False),
        76,
    ),
    (
        "cosine_backbone",
        lambda G: nb.cosine_backbone(G),
        lambda H: nb.fraction_filter(H, "cosine", 0.3, ascending=False),
        76,
    ),
    (
        "hub_promoted_index",
        lambda G: nb.hub_promoted_index(G),
        lambda H: nb.fraction_filter(H, "hpi", 0.3, ascending=False),
        76,
    ),
    (
        "hub_depressed_index",
        lambda G: nb.hub_depressed_index(G),
        lambda H: nb.fraction_filter(H, "hdi", 0.3, ascending=False),
        76,
    ),
    (
        "lhn_local_index",
        lambda G: nb.lhn_local_index(G),
        lambda H: nb.fraction_filter(H, "lhn_local", 0.3, ascending=False),
        76,
    ),
    (
        "preferential_attachment_score",
        lambda G: nb.preferential_attachment_score(G),
        lambda H: nb.fraction_filter(H, "pa", 0.3, ascending=False),
        76,
    ),
    (
        "adamic_adar_index",
        lambda G: nb.adamic_adar_index(G),
        lambda H: nb.fraction_filter(H, "adamic_adar", 0.3, ascending=False),
        76,
    ),
    (
        "resource_allocation_index",
        lambda G: nb.resource_allocation_index(G),
        lambda H: nb.fraction_filter(H, "resource_allocation", 0.3, ascending=False),
        76,
    ),
    (
        "graph_distance_proximity",
        lambda G: nb.graph_distance_proximity(G),
        lambda H: nb.fraction_filter(H, "dist", 0.3, ascending=False),
        76,
    ),
    (
        "local_path_index",
        lambda G: nb.local_path_index(G),
        lambda H: nb.fraction_filter(H, "lp", 0.3, ascending=False),
        76,
    ),
    (
        "glab_filter",
        lambda G: nb.glab_filter(G),
        lambda H: nb.threshold_filter(H, "glab_pvalue", DEFAULT_PVALUE, mode="below"),
        56,
    ),
]

UNWEIGHTED_METHOD_EDGE_COUNTS = [
    (
        "sparsify",
        lambda G: nb.sparsify(G),
        lambda H: nb.boolean_filter(H, "sparsify_keep"),
        174,
    ),
    (
        "lspar",
        lambda G: nb.lspar(G),
        lambda H: nb.boolean_filter(H, "sparsify_keep"),
        174,
    ),
    (
        "local_degree",
        lambda G: nb.local_degree(G),
        lambda H: nb.boolean_filter(H, "sparsify_keep"),
        91,
    ),
]


def test_les_miserables_baseline(les_miserables_weighted, les_miserables_unweighted):
    """NetworkX Les Miserables generator should remain stable."""
    assert les_miserables_weighted.number_of_nodes() == 77
    assert les_miserables_weighted.number_of_edges() == 254
    assert les_miserables_unweighted.number_of_edges() == 254


def test_warning_trigger_for_no_edge_reduction():
    with pytest.warns(UserWarning, match="same number of edges"):
        _warn_if_no_edge_reduction("dummy_method", 10, 10)


@pytest.mark.parametrize(
    ("method_name", "score_method", "filter_method", "expected_edges"),
    WEIGHTED_METHOD_EDGE_COUNTS,
    ids=[m[0] for m in WEIGHTED_METHOD_EDGE_COUNTS],
)
def test_weighted_methods_edge_counts(
    method_name,
    score_method,
    filter_method,
    expected_edges,
    les_miserables_weighted,
):
    """Weighted methods must use score-then-filter before benchmarking."""
    scored = score_method(les_miserables_weighted)
    filtered = filter_method(scored)
    assert isinstance(filtered, nx.Graph), method_name
    assert filtered.number_of_nodes() == les_miserables_weighted.number_of_nodes(), (
        method_name
    )
    assert filtered.number_of_edges() == expected_edges, method_name
    _warn_if_no_edge_reduction(
        method_name,
        les_miserables_weighted.number_of_edges(),
        filtered.number_of_edges(),
    )


@pytest.mark.parametrize(
    ("method_name", "score_method", "filter_method", "expected_edges"),
    UNWEIGHTED_METHOD_EDGE_COUNTS,
    ids=[m[0] for m in UNWEIGHTED_METHOD_EDGE_COUNTS],
)
def test_unweighted_methods_edge_counts(
    method_name,
    score_method,
    filter_method,
    expected_edges,
    les_miserables_unweighted,
):
    """Unweighted methods must use score-then-filter before benchmarking."""
    scored = score_method(les_miserables_unweighted)
    filtered = filter_method(scored)
    assert isinstance(filtered, nx.Graph), method_name
    assert filtered.number_of_nodes() == les_miserables_unweighted.number_of_nodes(), (
        method_name
    )
    assert filtered.number_of_edges() == expected_edges, method_name
    _warn_if_no_edge_reduction(
        method_name,
        les_miserables_unweighted.number_of_edges(),
        filtered.number_of_edges(),
    )
