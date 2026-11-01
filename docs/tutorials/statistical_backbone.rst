Statistical Backbone Extraction
================================

This tutorial demonstrates six statistical backbone extraction methods.
Each method tests whether an edge's weight is statistically significant under
a null model.

Setup
-----

We use a weighted graph for this tutorial::

    import networkx as nx
    import networkx_backbone as nb

    # Create a weighted graph
    G = nx.les_miserables_graph()

    print(f"Original: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

Disparity filter
----------------

The disparity filter (Serrano et al., 2009) tests each edge against a null
model where a node's total strength is uniformly distributed across its edges.
Edges with unexpectedly large weight receive low p-values::

    H = nb.disparity_filter(G)

    # Each edge now has a "disparity_pvalue" attribute
    u, v = list(H.edges())[0]
    print(f"Edge ({u}, {v}): p-value = {H[u][v]['disparity_pvalue']:.4f}")

    # Filter at alpha = 0.05
    backbone = nb.threshold_filter(H, "disparity_pvalue", 0.05)
    print(f"Disparity backbone: {backbone.number_of_edges()} edges")

Noise-corrected filter
----------------------

The noise-corrected filter (Coscia & Neffke, 2017) uses a binomial framework
to model edge weights. It produces z-scores rather than p-values -- higher
z-scores indicate more significant edges::

    H = nb.noise_corrected_filter(G)

    # Each edge now has an "nc_score" attribute (z-score)
    u, v = list(H.edges())[0]
    print(f"Edge ({u}, {v}): z-score = {H[u][v]['nc_score']:.4f}")

    # Filter: keep edges with z-score above a threshold
    backbone = nb.threshold_filter(H, "nc_score", 2.0, mode="above")
    print(f"Noise-corrected backbone: {backbone.number_of_edges()} edges")

Marginal likelihood filter
--------------------------

The marginal likelihood filter (Dianati, 2016) considers both endpoints in a
binomial null model and treats weights as integer counts::

    H = nb.marginal_likelihood_filter(G)
    backbone = nb.threshold_filter(H, "ml_pvalue", 0.05)
    print(f"Marginal likelihood backbone: {backbone.number_of_edges()} edges")

ECM filter
----------

The Enhanced Configuration Model (Gemmetto et al., 2017) uses a maximum-entropy
null model that preserves both degree and strength sequences. This is the most
principled null model but also the most computationally expensive::

    H = nb.ecm_filter(G)
    backbone = nb.threshold_filter(H, "ecm_pvalue", 0.05)
    print(f"ECM backbone: {backbone.number_of_edges()} edges")

LANS filter
-----------

Locally Adaptive Network Sparsification (Foti et al., 2011) uses nonparametric
empirical CDFs instead of parametric distributions. This makes no distributional
assumptions::

    H = nb.lans_filter(G)
    backbone = nb.threshold_filter(H, "lans_pvalue", 0.05)
    print(f"LANS backbone: {backbone.number_of_edges()} edges")

Multiple linkage analysis
-------------------------

Multiple linkage analysis (Van Nuffel et al., 2010; Yassin et al., 2023)
extracts a backbone using local linkage significance::

    H = nb.multiple_linkage_analysis(G, alpha=0.05)
    backbone = nb.boolean_filter(H, "mla_keep")
    print(f"MLA backbone: {backbone.number_of_edges()} edges")

Comparing all statistical methods
----------------------------------

Use :func:`~networkx_backbone.compare_backbones` to compare the results::

    backbones = {
        "disparity": nb.threshold_filter(
            nb.disparity_filter(G), "disparity_pvalue", 0.05
        ),
        "noise_corrected": nb.threshold_filter(
            nb.noise_corrected_filter(G), "nc_score", 2.0, mode="above"
        ),
        "marginal_likelihood": nb.threshold_filter(
            nb.marginal_likelihood_filter(G), "ml_pvalue", 0.05
        ),
        "ecm": nb.threshold_filter(
            nb.ecm_filter(G), "ecm_pvalue", 0.05
        ),
        "lans": nb.threshold_filter(
            nb.lans_filter(G), "lans_pvalue", 0.05
        ),
        "mla": nb.boolean_filter(
            nb.multiple_linkage_analysis(G, alpha=0.05), "mla_keep"
        ),
    }

    results = nb.compare_backbones(G, backbones)
    for name, metrics in results.items():
        ef = metrics["edge_fraction"]
        nf = metrics["node_fraction"]
        print(f"{name:25s}: edges={ef:.1%}, nodes={nf:.1%}")

Multiple-testing correction
---------------------------

Because a statistical backbone tests every edge, correcting for multiple
comparisons avoids retaining edges by chance. Pass ``mtc`` to
:func:`~networkx_backbone.threshold_filter` (or use
:func:`~networkx_backbone.adjust_pvalues` directly). Supported corrections match
R's ``p.adjust``: ``"bonferroni"``, ``"holm"``, ``"hochberg"``, ``"bh"``/``"fdr"``
(Benjamini-Hochberg), and ``"by"`` (Benjamini-Yekutieli)::

    scored = nb.disparity_filter(G)

    raw = nb.threshold_filter(scored, "disparity_pvalue", 0.05)
    fdr = nb.threshold_filter(scored, "disparity_pvalue", 0.05, mtc="bh")
    bonf = nb.threshold_filter(scored, "disparity_pvalue", 0.05, mtc="bonferroni")

    print(f"raw={raw.number_of_edges()}, BH={fdr.number_of_edges()}, "
          f"Bonferroni={bonf.number_of_edges()}")

Signed backbones
----------------

By default the statistical filters run a one-tailed test that keeps only
*significantly strong* edges. With ``signed=True``, the
``disparity_filter``, ``marginal_likelihood_filter``, and ``lans_filter`` run a
**two-tailed** test: the stored p-value becomes two-sided, and each edge gains a
``"sign"`` attribute that is ``+1`` for a significantly strong edge and ``-1``
for a significantly weak one::

    signed = nb.disparity_filter(G, signed=True)
    backbone = nb.threshold_filter(signed, "disparity_pvalue", 0.05, mtc="bh")

    positive = [(u, v) for u, v, d in backbone.edges(data=True) if d["sign"] == 1]
    negative = [(u, v) for u, v, d in backbone.edges(data=True) if d["sign"] == -1]
    print(f"strong (+): {len(positive)}, weak (-): {len(negative)}")

References
----------

- Serrano, M. A., Boguna, M., & Vespignani, A. (2009). *Extracting the multiscale
  backbone of complex weighted networks*. PNAS, 106(16), 6483-6488.
- Coscia, M., & Neffke, F. M. (2017). *Network backboning with noisy data*.
  Proc. IEEE ICDE, 425-436.
- Dianati, N. (2016). *Unwinding the hairball graph: Pruning algorithms for
  weighted complex networks*. Physical Review E, 93, 012304.
- Gemmetto, V., Cardillo, A., & Garlaschelli, D. (2017). *Irreducible network
  backbones: unbiased graph filtering via maximum entropy*. arXiv:1706.00230.
- Foti, N. J., Hughes, J. M., & Rockmore, D. N. (2011). *Nonparametric
  sparsification of complex multiscale networks*. PLOS ONE, 6(2), e16431.
- Van Nuffel, N., Saey, P., Derudder, B., Devriendt, L., & Witlox, F. (2010).
  *Measuring hierarchical differentiation: connectivity and dominance in the
  European urban network*. Transportation Planning and Technology, 33(4),
  343-366.
- Benjamini, Y., & Hochberg, Y. (1995). *Controlling the false discovery rate*.
  J. Royal Statistical Society B, 57(1), 289-300.
- Benjamini, Y., & Yekutieli, D. (2001). *The control of the false discovery rate
  in multiple testing under dependency*. Annals of Statistics, 29(4), 1165-1188.
