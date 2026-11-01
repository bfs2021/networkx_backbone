Les Miserables Benchmark
========================

This benchmark uses NetworkX's built-in ``nx.les_miserables_graph()`` and
applies a strict **score-then-filter** workflow for every method:

1. Run a scoring method that returns the full graph with edge attributes.
2. Apply a filter (`threshold_filter`, `fraction_filter`, or `boolean_filter`).
3. Report edge counts only from the filtered graph.

If the filtered graph has the same edge count as the original graph, issue a
warning and re-test the method parameters.

Setup
-----

::

    import warnings
    import networkx as nx
    import networkx_backbone as nb

    G = nx.les_miserables_graph()
    G_unweighted = nx.Graph()
    G_unweighted.add_nodes_from(G.nodes(data=True))
    G_unweighted.add_edges_from(G.edges())

    ORIGINAL_EDGES = G.number_of_edges()  # 254

    def warn_if_no_reduction(name, filtered):
        if filtered.number_of_edges() == ORIGINAL_EDGES:
            warnings.warn(
                f"{name}: filtered graph has the same number of edges as "
                f"the original ({ORIGINAL_EDGES}). Re-test and validate.",
                UserWarning,
            )

Filtered Edge Counts
--------------------

All counts below are after explicit filtering.

.. list-table::
   :header-rows: 1
   :widths: 48 26 26

   * - Function
     - Filter Applied
     - Edges After Filter
   * - :func:`~networkx_backbone.disparity_filter`
     - ``disparity_pvalue < 0.05``
     - 9
   * - :func:`~networkx_backbone.noise_corrected_filter`
     - ``nc_score >= 2.0``
     - 98
   * - :func:`~networkx_backbone.marginal_likelihood_filter`
     - ``ml_pvalue < 0.05``
     - 126
   * - :func:`~networkx_backbone.ecm_filter`
     - ``ecm_pvalue < 0.05``
     - 72
   * - :func:`~networkx_backbone.lans_filter`
     - ``lans_pvalue < 0.05``
     - 109
   * - :func:`~networkx_backbone.multiple_linkage_analysis`
     - ``mla_keep`` (boolean)
     - 109
   * - :func:`~networkx_backbone.global_threshold_filter`
     - ``global_threshold_keep`` (boolean)
     - 157
   * - :func:`~networkx_backbone.strongest_n_ties`
     - ``strongest_n_ties_keep`` (boolean)
     - 113
   * - :func:`~networkx_backbone.global_sparsification`
     - ``global_sparsification_keep`` (boolean)
     - 127
   * - :func:`~networkx_backbone.primary_linkage_analysis`
     - ``primary_linkage_keep`` (boolean)
     - 69
   * - :func:`~networkx_backbone.edge_betweenness_filter`
     - ``edge_betweenness_keep`` (boolean)
     - 127
   * - :func:`~networkx_backbone.node_degree_filter`
     - ``node_degree_keep`` (boolean)
     - 237
   * - :func:`~networkx_backbone.high_salience_skeleton`
     - ``salience >= 0.5``
     - 76
   * - :func:`~networkx_backbone.metric_backbone`
     - ``metric_keep`` (boolean)
     - 163
   * - :func:`~networkx_backbone.ultrametric_backbone`
     - ``ultrametric_keep`` (boolean)
     - 118
   * - :func:`~networkx_backbone.doubly_stochastic_filter`
     - ``ds_weight >= 0.1``
     - 109
   * - :func:`~networkx_backbone.h_backbone`
     - ``h_backbone_keep`` (boolean)
     - 18
   * - :func:`~networkx_backbone.modularity_backbone`
     - ``modularity_keep`` (boolean, ``seed=42``)
     - 72
   * - :func:`~networkx_backbone.planar_maximally_filtered_graph`
     - ``pmfg_keep`` (boolean)
     - 162
   * - :func:`~networkx_backbone.maximum_spanning_tree_backbone`
     - ``mst_keep`` (boolean)
     - 76
   * - :func:`~networkx_backbone.neighborhood_overlap`
     - top 30% by ``overlap``
     - 76
   * - :func:`~networkx_backbone.jaccard_backbone`
     - top 30% by ``jaccard``
     - 76
   * - :func:`~networkx_backbone.dice_backbone`
     - top 30% by ``dice``
     - 76
   * - :func:`~networkx_backbone.cosine_backbone`
     - top 30% by ``cosine``
     - 76
   * - :func:`~networkx_backbone.hub_promoted_index`
     - top 30% by ``hpi``
     - 76
   * - :func:`~networkx_backbone.hub_depressed_index`
     - top 30% by ``hdi``
     - 76
   * - :func:`~networkx_backbone.lhn_local_index`
     - top 30% by ``lhn_local``
     - 76
   * - :func:`~networkx_backbone.preferential_attachment_score`
     - top 30% by ``pa``
     - 76
   * - :func:`~networkx_backbone.adamic_adar_index`
     - top 30% by ``adamic_adar``
     - 76
   * - :func:`~networkx_backbone.resource_allocation_index`
     - top 30% by ``resource_allocation``
     - 76
   * - :func:`~networkx_backbone.graph_distance_proximity`
     - top 30% by ``dist``
     - 76
   * - :func:`~networkx_backbone.local_path_index`
     - top 30% by ``lp``
     - 76
   * - :func:`~networkx_backbone.glab_filter`
     - ``glab_pvalue < 0.05``
     - 56
   * - :func:`~networkx_backbone.sparsify`
     - ``sparsify_keep`` (boolean)
     - 174
   * - :func:`~networkx_backbone.lspar`
     - ``sparsify_keep`` (boolean)
     - 174
   * - :func:`~networkx_backbone.local_degree`
     - ``sparsify_keep`` (boolean)
     - 91

Reproducibility Script
----------------------

::

    import warnings
    import networkx as nx
    import networkx_backbone as nb

    G = nx.les_miserables_graph()
    G_unweighted = nx.Graph()
    G_unweighted.add_nodes_from(G.nodes(data=True))
    G_unweighted.add_edges_from(G.edges())

    original_edges = G.number_of_edges()

    methods = [
        ("disparity_filter", lambda: nb.threshold_filter(nb.disparity_filter(G), "disparity_pvalue", 0.05, mode="below")),
        ("noise_corrected_filter", lambda: nb.threshold_filter(nb.noise_corrected_filter(G), "nc_score", 2.0, mode="above")),
        ("marginal_likelihood_filter", lambda: nb.threshold_filter(nb.marginal_likelihood_filter(G), "ml_pvalue", 0.05, mode="below")),
        ("ecm_filter", lambda: nb.threshold_filter(nb.ecm_filter(G), "ecm_pvalue", 0.05, mode="below")),
        ("lans_filter", lambda: nb.threshold_filter(nb.lans_filter(G), "lans_pvalue", 0.05, mode="below")),
        ("multiple_linkage_analysis(alpha=0.05)", lambda: nb.boolean_filter(nb.multiple_linkage_analysis(G, alpha=0.05), "mla_keep")),
        ("global_threshold_filter(threshold=2)", lambda: nb.boolean_filter(nb.global_threshold_filter(G, threshold=2), "global_threshold_keep")),
        ("strongest_n_ties(n=2)", lambda: nb.boolean_filter(nb.strongest_n_ties(G, n=2), "strongest_n_ties_keep")),
        ("global_sparsification(s=0.5)", lambda: nb.boolean_filter(nb.global_sparsification(G, s=0.5), "global_sparsification_keep")),
        ("primary_linkage_analysis", lambda: nb.boolean_filter(nb.primary_linkage_analysis(G), "primary_linkage_keep")),
        ("edge_betweenness_filter(s=0.5)", lambda: nb.boolean_filter(nb.edge_betweenness_filter(G, s=0.5), "edge_betweenness_keep")),
        ("node_degree_filter(min_degree=2)", lambda: nb.boolean_filter(nb.node_degree_filter(G, min_degree=2), "node_degree_keep")),
        ("high_salience_skeleton", lambda: nb.threshold_filter(nb.high_salience_skeleton(G), "salience", 0.5, mode="above")),
        ("metric_backbone", lambda: nb.boolean_filter(nb.metric_backbone(G), "metric_keep")),
        ("ultrametric_backbone", lambda: nb.boolean_filter(nb.ultrametric_backbone(G), "ultrametric_keep")),
        ("doubly_stochastic_filter", lambda: nb.threshold_filter(nb.doubly_stochastic_filter(G), "ds_weight", 0.1, mode="above")),
        ("h_backbone", lambda: nb.boolean_filter(nb.h_backbone(G), "h_backbone_keep")),
        ("modularity_backbone", lambda: nb.boolean_filter(nb.modularity_backbone(G, seed=42), "modularity_keep")),
        ("planar_maximally_filtered_graph", lambda: nb.boolean_filter(nb.planar_maximally_filtered_graph(G), "pmfg_keep")),
        ("maximum_spanning_tree_backbone", lambda: nb.boolean_filter(nb.maximum_spanning_tree_backbone(G), "mst_keep")),
        ("neighborhood_overlap", lambda: nb.fraction_filter(nb.neighborhood_overlap(G), "overlap", 0.3, ascending=False)),
        ("jaccard_backbone", lambda: nb.fraction_filter(nb.jaccard_backbone(G), "jaccard", 0.3, ascending=False)),
        ("dice_backbone", lambda: nb.fraction_filter(nb.dice_backbone(G), "dice", 0.3, ascending=False)),
        ("cosine_backbone", lambda: nb.fraction_filter(nb.cosine_backbone(G), "cosine", 0.3, ascending=False)),
        ("hub_promoted_index", lambda: nb.fraction_filter(nb.hub_promoted_index(G), "hpi", 0.3, ascending=False)),
        ("hub_depressed_index", lambda: nb.fraction_filter(nb.hub_depressed_index(G), "hdi", 0.3, ascending=False)),
        ("lhn_local_index", lambda: nb.fraction_filter(nb.lhn_local_index(G), "lhn_local", 0.3, ascending=False)),
        ("preferential_attachment_score", lambda: nb.fraction_filter(nb.preferential_attachment_score(G), "pa", 0.3, ascending=False)),
        ("adamic_adar_index", lambda: nb.fraction_filter(nb.adamic_adar_index(G), "adamic_adar", 0.3, ascending=False)),
        ("resource_allocation_index", lambda: nb.fraction_filter(nb.resource_allocation_index(G), "resource_allocation", 0.3, ascending=False)),
        ("graph_distance_proximity", lambda: nb.fraction_filter(nb.graph_distance_proximity(G), "dist", 0.3, ascending=False)),
        ("local_path_index", lambda: nb.fraction_filter(nb.local_path_index(G), "lp", 0.3, ascending=False)),
        ("glab_filter", lambda: nb.threshold_filter(nb.glab_filter(G), "glab_pvalue", 0.05, mode="below")),
        ("sparsify", lambda: nb.boolean_filter(nb.sparsify(G_unweighted), "sparsify_keep")),
        ("lspar", lambda: nb.boolean_filter(nb.lspar(G_unweighted), "sparsify_keep")),
        ("local_degree", lambda: nb.boolean_filter(nb.local_degree(G_unweighted), "sparsify_keep")),
    ]

    print("original_edges", original_edges)
    for name, fn in methods:
        filtered = fn()
        if filtered.number_of_edges() == original_edges:
            warnings.warn(
                f"{name}: filtered graph has the same number of edges as "
                f"the original ({original_edges}). Re-test and validate.",
                UserWarning,
            )
        print(name, filtered.number_of_edges())
