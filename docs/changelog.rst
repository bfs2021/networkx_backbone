Changelog
=========

Unreleased (0.2.0)
------------------

This release adds hypergraph backbones and the remaining Backbone 3.0
features. It also fixes many correctness problems found in a full code review
that checked every method against its paper. **Several methods now return
different results**; see *Changes in results* below. Remaining review findings
are tracked in the :doc:`roadmap` and in GitHub issues #9-#31.

**New features**

- Hypergraph module: the parameter-free MDL hypergraph backbone (Kirkley et
  al., 2026) with ``"edge"``, ``"node"``, and ``"auto"`` optimizers;
  statistically validated hypergraphs and cores (SVH/SVC); maximal
  hyperedges, order filtering, s-components, and intersection graphs.
- ``hypergraph_io``: ``hypergraph_to_bipartite``, the HIF interchange format,
  and adapters for xgi, HyperNetX, HypergraphX, and HAT.
- Backbone 3.0 parity: multiple-testing correction
  (:func:`~networkx_backbone.adjust_pvalues`, ``threshold_filter(mtc=...)``),
  signed backbones (``signed=True``), and SDSM-EC edge constraints.
- ``backbone_from_projection(mtc=...)``; ``sdsm(pvalue_method=...)``;
  ``fdsm(burn_in=..., thin=...)``; ``bicm(tol=..., max_iter=...)``.
- ``seed=`` on ``modularity_backbone``, ``sparsify``, and ``compare_graphs``.
- ``multigraph_to_weighted(aggregate=...)``, which can sum, average, or take
  the max/min of parallel-edge weights.
- ``local_path_index(weight=...)``; a ``h_bridge`` attribute from
  ``h_backbone``; ``statistically_validated_cores(max_groups=...)``.

**Changes in results (bug fixes)**

The Les Misérables counts below are edges kept at the documented threshold.
Davis counts are significant woman-woman pairs at alpha = 0.05.

- ``ecm_filter``: the solver never converged and every p-value was about 0,
  so every edge was kept (254 of 254). It now fits the ECM by maximum
  likelihood and uses the correct geometric tail; it keeps 72 edges and warns
  if the fit does not converge.
- ``marginal_likelihood_filter``: now uses Dianati's null,
  :math:`B(T, s_is_j/2T^2)`. The old formula depended on the order nodes were
  added. Les Misérables: 70 -> 126 edges.
- ``glab_filter``: now uses per-node shortest-path involvement and
  :math:`(1-I_{ij})^{(k_i-1)^c}` (Zhang et al., 2014) instead of global edge
  betweenness. Les Misérables: 5 -> 56 edges.
- ``disparity_filter`` on directed graphs also tests the target's incoming
  links.
- ``bicm``/``sdsm``: the BiCM is now actually fitted rather than
  approximated with Chung-Lu, and p-values use exact Poisson-binomial tails
  (or RNA). SDSM-EC refits on the free cells. Davis: 3 -> 0 pairs.
- ``fdsm``: the null now preserves degrees (a curveball chain; previously
  197 of 200 samples broke the margins), and p-values are
  ``(count + 1) / (trials + 1)``. Davis: about 10 -> 5 pairs.
- ``fixedfill`` and ``fixedcol`` use their exact null distributions.
- ``backbone_from_projection`` and ``backbone()`` now threshold ``sdsm`` and
  ``fdsm`` at ``alpha``, as they already did for the other models.
- ``lspar``/``sparsify`` and ``local_degree``: each node ranks its own edges
  (Satuluri et al., 2011; Lindner et al., 2015), and ties are kept together.
  Les Misérables: 136 -> 174 (LSpar) and 135 -> 91 (local degree).
- ``h_backbone`` follows Zhang et al. (2018): unweighted betweenness on the
  full graph, normalized by N. When h = 0 no edges are kept. Les Misérables:
  22 -> 18.
- ``high_salience_skeleton`` splits credit among tied shortest paths, so
  saliences sum to n - 1.
- ``metric_backbone`` uses a relative tolerance; ``ultrametric_backbone`` no
  longer crashes on self-loops.
- Proximity indices exclude self-loops from neighborhoods and use out-degree
  for directed graphs.
- ``doubly_stochastic_filter`` warns when Sinkhorn-Knopp does not converge and
  no longer accepts a loose convergence check.
- ``reachability`` counts directed reachability correctly;
  ``compare_backbones`` forwards ``weight`` to every measure that accepts it.
- ``adjust_pvalues`` keeps NaN as NaN and excludes it from the number of tests
  (as R's ``p.adjust`` does), instead of making it significant.

**API conventions (NetworkX compatibility)**

- Every scorer raises ``NetworkXNotImplemented`` for multigraphs (use
  :func:`~networkx_backbone.multigraph_to_weighted` first), and
  ``NetworkXError`` for missing, non-positive, NaN, or infinite weights.
- Bipartite functions accept generators for ``agent_nodes``, reject agent sets
  that are not one side of the bipartition, and raise ``AmbiguousSolution``
  when the partition cannot be inferred.
- Filters keep graph, node, and edge attributes and multigraph keys.
  ``fraction_filter`` ignores unscored elements, and ``consensus_backbone``
  works with node labels of mixed types and returns edges in a deterministic
  order.
- Functions that used a hardcoded ``seed=42`` now default to ``seed=None``,
  following NetworkX. Pass a seed for reproducible results.

**Performance**

- Vectorized ``sdsm``/``fixed*``/``fdsm`` (``sdsm`` on 400x300: 6.4 s ->
  0.36 s); MDL ``"node"`` optimizer with a lazy heap (m = 2000: 8.3 s ->
  0.47 s); vectorized SVC p-values.

**Infrastructure and documentation**

- CI on every push and pull request: ruff lint and format checks, tests and
  doctests on Python 3.10-3.14, and a NetworkX-3.0-only job that checks the
  optional dependencies stay optional. Test coverage is 97%.
- New :doc:`roadmap`; full numpydoc sections for the bipartite, hypergraph,
  and structural APIs; corrected citations (Van Nuffel et al., 2010; Lü et
  al., 2009; Lindner et al., 2015); the docs version is taken from the package
  metadata.


Version 0.1.0 (Initial Release)
--------------------------------

First public release of ``networkx-backbone``.

**Features:**

- 47 backbone extraction functions across 8 modules
- Statistical methods: disparity filter, noise-corrected filter, marginal
  likelihood filter, ECM filter, LANS filter
- Structural methods: global threshold, strongest N ties, high salience
  skeleton, metric backbone, ultrametric backbone, doubly stochastic filter,
  h-backbone, modularity backbone, planar maximally filtered graph, maximum
  spanning tree backbone
- Proximity methods: neighborhood overlap, Jaccard, Dice, cosine, hub
  promoted/depressed indices, LHN local index, preferential attachment,
  Adamic-Adar, resource allocation, graph distance proximity, local path index
- Hybrid methods: GLAB filter
- Bipartite methods: SDSM, FDSM
- Unweighted sparsification: sparsify, LSpar, local degree
- Filtering utilities: threshold, fraction, boolean, consensus backbone
- Evaluation measures: node/edge/weight fractions, reachability, KS degree,
  KS weight, compare backbones
