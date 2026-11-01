Roadmap
=======

This page tracks where ``networkx-backbone`` is headed: what was finished
recently, known gaps against the cited methods, and planned features. It
consolidates the design notes in ``docs/design/`` (the
`Backbone 3.0 coverage evaluation
<https://github.com/brianckeegan/networkx_backbone/blob/main/docs/design/backbone-3.0-coverage.md>`_
and the `hypergraph backboning proposal
<https://github.com/brianckeegan/networkx_backbone/blob/main/docs/design/hypergraph-backboning.md>`_).
Contributions to any item are welcome. Please open an issue first for the
larger ones.

Status key: **done**, **next** (planned for the next release), **planned**,
**idea** (open for discussion).

Recently completed
------------------

Method coverage
~~~~~~~~~~~~~~~

- **done**: Every backbone *model* in Neal's Backbone 3.0 R package, plus its
  cross-cutting features: multiple-testing correction (``mtc``), signed
  backbones, and SDSM-EC edge constraints.
- **done**: Hypergraph backbones: the MDL backbone (Kirkley et al., 2026),
  statistically validated hypergraphs and cores (SVH/SVC), structural
  reductions, and HIF/xgi/HyperNetX/HypergraphX/HAT interoperability.

Correctness review (2026-11)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A full code review checked each method against its paper. The fixes below
change numerical results, so they are listed in the :doc:`changelog` as
breaking changes:

- ``ecm_filter``: the solver did not converge and every p-value was about 0.
  It is now a maximum-likelihood fit with a Newton polish, using the geometric
  tail :math:`p_{ij}(y_iy_j)^{w-1}`.
- ``marginal_likelihood_filter``: now Dianati's null model
  :math:`B(T, s_is_j/2T^2)`. The old one was asymmetric and depended on the
  order nodes were added.
- ``glab_filter``: now uses per-node shortest-path involvement and
  :math:`(1-I_{ij})^{(k_i-1)^c}` (Zhang et al., 2014) instead of global edge
  betweenness.
- ``disparity_filter`` on directed graphs: now also tests the target's
  incoming links (Serrano et al., 2009).
- The FDSM null now preserves degrees (a curveball chain), and the BiCM is
  fitted exactly. See the changelog for the full bipartite and structural
  lists.
- Shared conventions: every scorer rejects multigraphs, rejects NaN or
  infinite weights, works with node labels of mixed types, and preserves
  graph, node, and edge attributes through the filters.
- CI now runs tests on every push and pull request (Python 3.10-3.14 and a
  NetworkX-only job), along with ruff lint and format checks and doctests.

Next release (0.2.0)
--------------------

Correctness gaps found in review and not yet fixed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **next**: ``noise_corrected_filter`` is a binomial z-score. Implement the
  full Bayesian noise-corrected backbone of Coscia & Neffke (2017): the
  posterior variance from the beta prior plus the delta method, and the
  transformed lift score. Keep the current score as an option for backward
  compatibility.
- **next**: ``multiple_linkage_analysis`` currently thresholds LANS p-values
  (the netbone convention). Implement the dominance-fitting procedure of
  Van Nuffel et al. (2010) and keep the netbone behavior behind an option.
- **next**: Directed ``lans_filter`` only uses the source's out-links. Also
  test the target's in-links, as ``disparity_filter`` now does.
- **next**: A directed GLAB variant, using the out- and in-involvement of
  Zhang et al. (2014), Section 2.3.
- **next**: ``fixedfill``: replace the ``Binomial(n, fill²)`` approximation
  with the exact fixed-fill null distribution of Neal et al. (2021).
- **next**: Check ``hyper_projection``'s weighting (:math:`1/d(z)` versus
  Newman's :math:`1/(d(z)-1)`) and the arXiv identifier cited for the
  projection methods against Coscia & Neffke (2017).

Validation
~~~~~~~~~~

- **next**: Cross-implementation reference tests. Pin p-values on shared
  datasets (Les Misérables, Davis Southern Women, karate) against R
  ``backbone`` 3.0 and ``netbone``, stored as small JSON fixtures so the test
  suite needs neither R nor netbone.
- **next**: An optional CI job that installs xgi, HyperNetX, HypergraphX, and
  HAT and runs the adapter round-trip tests against the real libraries; they
  currently run only against stubs.
- **planned**: Turn on numpydoc validation (``numpydoc_validation_checks``)
  in the docs build so missing Parameters, Returns, or Examples sections fail
  CI.

Release engineering
~~~~~~~~~~~~~~~~~~~

- **next**: Tag ``v0.2.0``. The repository has no version tags yet, so
  ``setuptools-scm`` falls back to ``0.0.0``. Publish to PyPI and conda-forge
  from the existing workflows.
- **planned**: Add ``py.typed`` and type annotations for the public API.

Planned features
----------------

API and ecosystem compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **planned**: NetworkX backend dispatching (``@nx._dispatchable``) so
  backends such as ``nx-cugraph`` and ``nx-parallel`` can accelerate the
  shortest-path-heavy methods (salience, metric backbone, GLAB).
- **planned**: Accept SciPy sparse arrays and pandas edge lists as input to the
  statistical scorers, with ``to_*`` helpers back to NetworkX, matching how
  NetworkX interoperates with the NumPy/SciPy/pandas stack.
- **planned**: Make the bipartite and projection code fully sparse (no dense
  agent-by-artifact matrices) and vectorize the remaining per-pair loops.
- **planned**: A sparse or blockwise ECM solver for graphs with more than
  about 2,000 nodes, where the dense Newton polish is skipped.
- **idea**: An ``asv`` benchmark suite, so performance regressions show up in
  review.

Methods
~~~~~~~

- **planned**: ``describe_backbone`` (Backbone 3.0's ``narrative``): methods
  text and citations for a chosen backbone and its parameters.
- **planned**: The local (per-node-neighborhood) MDL hypergraph backbone
  (Kirkley et al., 2026, Appendix E).
- **planned**: Signed variants of ``ecm_filter`` and the noise-corrected
  filter, for parity with the other scorers' ``signed=True``.
- **idea**: A ``missing_as_zero`` toggle for weighted inputs (Backbone 3.0),
  for testing absent dyads.
- **idea**: Further methods from the Yassin et al. (2025) structural and
  statistical comparisons that are not covered yet. Candidates are tracked
  in GitHub issues.

Documentation
~~~~~~~~~~~~~

- **planned**: A "validating a backbone" tutorial showing the
  ``compare_backbones``, ``consensus_backbone``, and ``mtc`` workflow end to
  end.
- **planned**: Turn the Les Misérables benchmark table into an executed
  Sphinx-Gallery example, so its edge counts cannot go stale.
