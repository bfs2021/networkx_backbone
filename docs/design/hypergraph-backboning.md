# Design proposal: hypergraph network backboning

- **Status:** Draft / for discussion
- **Tracking:** open items now live in `docs/roadmap.rst` and GitHub issue #31
- **Scope:** Evaluation + proposed integration plan (no library code changed by this document)
- **Trigger:** Request to evaluate incorporating hypergraph network backboning
  methods into `networkx-backbone`, specifically the method of *Kirkley, Felippe,
  Malizia & Battiston, "Hypergraph backboning," arXiv:2606.00893 (2026)*, with
  optional interoperability for `xgi`, `hypergraphx`, `HyperNetX`, and the
  `Hypergraph-Analysis-Toolbox`, and an assessment of which hypergraph backbone
  methods are genuinely distinct from the existing ensemble.

> **Revision note (after reading the paper).** An earlier draft of this document
> was written without access to arXiv:2606.00893 and hypothesized that the paper
> belonged to the *statistical* hyperedge-filtering family (SVH/SVC). The paper
> has now been read in full: it is a **parameter-free, information-theoretic
> (MDL) compression** method — a *different paradigm* from statistical
> null-model testing. Sections 2, 6, and 9 are updated accordingly; the headline
> method to port is now the MDL backbone, with statistical/structural methods
> repositioned as complementary.

---

## 1. Summary and recommendation

**Recommendation: yes — port the paper's MDL hypergraph backbone as the headline
method, add optional interoperability with the major hypergraph libraries
(centered on the HIF interchange format), and offer a small set of complementary
hypergraph-native methods.** The case is now stronger than in the first draft: the
method is principled, **parameter-free** (unweighted), **handles weights** (the
paper's main novelty — no prior hypergraph filter did), needs only
`networkx` + `numpy`/`scipy`, and runs in minutes on real data.

Hypergraph backboning organizes into three paradigms:

| Paradigm | What it does | Output | Library status |
|----------|--------------|--------|----------------|
| **A — Projection backboning** | Hypergraph → weighted pairwise graph → null-model edge test | a graph | **already works** (verified, §5) |
| **B1 — Statistical hyperedge filtering** | Keep hyperedges over-expressed vs a null model (needs significance level α) | sub-hypergraph | not present (optional, §6) |
| **B2 — Information-theoretic (MDL) backboning ← this paper** | Compress nested/redundant hyperedges; keep the minimal "parent" set | sub-hypergraph | **port this** (§2, §9) |

Key conclusions:

1. **Family A already works today** with the existing `bipartite` module (§5).
2. **Most weighted-hypergraph backbones are generalizations** of methods we ship
   and become reachable for free once a hypergraph can enter the pipeline (§6.1).
3. **The paper's MDL method is genuinely sui generis** — it exploits overlap and
   nestedness, structural signatures unique to hypergraphs, via a global
   compression objective with no dyadic analog and no significance parameter
   (§2, §6.2). It is the right headline method to port.
4. **Optional interop is cheap** because all four target libraries converge on an
   incidence/bipartite substrate and on the **HIF** JSON format; none needs to
   become a *required* dependency (§7).

Phasing (details in §9): **Phase 0** ingestion (HIF + per-library adapters) and
surfacing of Family A; **Phase 1** the MDL backbone (unweighted + weighted);
**Phase 2** complementary statistical (SVH/SVC) and structural (toplex, s-line)
methods. Out of scope: hypergraph neural networks; any required third-party
hypergraph dependency.

## 2. The source paper (arXiv:2606.00893)

*A. Kirkley, H. Felippe, F. Malizia, F. Battiston, "Hypergraph backboning" (2026).*

**Idea.** Given a hypergraph `G` on `N` nodes (undirected hyperedges, no repeated
nodes within an edge, no multi-edges) with `L` distinct hyperedge sizes (orders),
find a backbone `B ⊆ G`: a subset of hyperedges ("parents") such that every
non-backbone hyperedge ("child") can be cheaply reconstructed from a parent it
overlaps. The best backbone is the one that **minimizes a two-part description
length** (MDL) — equivalently, that maximizes the structural redundancy
(overlap/nestedness) explained. It is **fully nonparametric** for unweighted
hypergraphs.

**Unweighted objective.** With `log ≡ log2` and `C(n,k)` the binomial:

- Transmit each parent `p ∈ B`:  `H(p) = log L + log C(N, |p|)`;  `L(B) = Σ_p H(p)`.
- Transmit each child `c` from its parent `p` (with `|p ∩ c| ≥ 1`):
  `H(c|p) = log L + log min(|p|,|c|) + log C(|p|, |p∩c|) + log C(N−|p|, |c|−|p∩c|)`.
- Total: `L(G,B) = L(B) + Σ_{p∈B} Σ_{c∈∂p} H(c|p)`, where `∂p` are the children of
  `p`.  The optimum is `B* = argmin_B L(G,B)`.
- Equivalent reduced-mutual-information form: `L(G,B) = L(G,G) − Σ_c R(c, p(c))`,
  so minimizing description length = **maximizing parent–child overlap/nestedness**.
- **Inverse compression ratio** `η = L(G,B*) / L(G,G) ∈ [0,1]` measures how
  compressible (redundant) the hypergraph is (`η→0` very compressible, `η=1` none).

**Weighted extension.** `Lw(G,B) = L(G,B) + Σ_e L(w(e), b_e)`, where `b_e∈{0,1}` is
backbone membership and weights follow a role-dependent **Poisson or Geometric**
prior under an empirical-Bayes mean constraint. A single hyperparameter
`γ ∈ (0,1]` trades off weight vs. topology: `γ=1` recovers the unweighted
objective (weights ignored); `γ→0` makes it infinitely costly to leave a
high-weight edge out of the backbone. The backbone-inclusion reward is **linear in
the weight** `w(e)`, with a closed-form weight threshold `w*` below which weight no
longer favors inclusion. (Integer weights `≥1`; continuous weights need a
resolution parameter.) **No prior hypergraph filtering method handled weights** —
this is the paper's central novelty and aligns with this library's weighted focus.

**Optimization (Appendix D).** Exact minimization is combinatorial; the paper uses
greedy approximations on the **intersection graph** `Int(G)` (one node per
hyperedge; link two hyperedges that share ≥1 node). Parent–child assignments form a
**partition of `Int(G)` into disjoint stars** (each child has exactly one parent;
parents/children don't nest further). Two greedy schemes — "node" addition and
"edge" addition (the latter usually better) — are run and the lower description
length is kept. Greedy compression is **indistinguishable from exact** on small
samples.

**Complexity / cost.** Bottleneck is building `Int(G)`: `O(Σ_i |G_i|²)` over node
neighborhoods `G_i = {e : i ∈ e}` (best case `O(N)`, worst `O(N|G|²)`); optional
random pair sampling gives `O(N s²)`. Empirically ~`N^1.17`, **≤6 minutes** on the
empirical corpus with a plain Python implementation. A **local variant**
(Appendix E) backbones each node neighborhood separately.

**Inputs / outputs.** Input: a hyperedge list over `N` nodes (+ optional integer
weights). Output: the backbone sub-hypergraph `B`, the parent→children assignment
(star forest), and `η`.

**Dependencies implied.** Only a hyperedge list and arithmetic; `Int(G)` is an
ordinary graph (build with NetworkX), and the log-binomials use
`scipy.special.gammaln`. **No third-party hypergraph library is required to
implement it.** No public reference code is cited (the authors describe a "simple
Python implementation"; datasets come from the Hypergraphx-data repository).

## 3. Background: three paradigms

- **A — projection backboning.** Represent `G` as an incidence (node × hyperedge)
  structure, project to a node–node weighted graph, apply a degree-preserving null
  model. Output is a graph. Lineage: Neal's `backbone` (Backbone 3.0, 2026),
  Coscia & Neffke (2017).
- **B1 — statistical hyperedge filtering.** Keep hyperedges over-expressed vs a
  configuration null, *given a significance level α*. Output is a sub-hypergraph.
  Reference: Musciotto, Battiston & Mantegna (2021); impl in HGX (`get_svh`/`get_svc`).
- **B2 — information-theoretic (MDL) backboning — this paper.** Compress nested and
  redundant hyperedges; keep the minimal parent set. Output is a sub-hypergraph.
  **Parameter-free** (unweighted), weighted via one knob. Distinct from B1: it is a
  global compression optimum, not a per-hyperedge hypothesis test, and needs no α.

Families B1/B2 share the "sub-hypergraph output" problem that does not fit the
current graph-in/graph-out filters.

## 4. Current library capabilities relevant to this

- **Two idioms already in use:** *score-then-filter* (e.g. `disparity_filter` →
  `threshold_filter`) and *direct boolean flag* (e.g.
  `maximum_spanning_tree_backbone` → `boolean_filter` on `mst_keep`). The MDL
  method is a global optimizer, so it maps onto the **boolean-flag idiom** (annotate
  each hyperedge with an `mdl_keep` role), not onto per-edge p-value thresholding.
- **Dependency policy:** core `networkx`-only; `numpy`/`scipy` are the optional
  `[full]` extra, imported lazily. The MDL method fits this exactly.
- **The `bipartite` module is already an incidence engine** (`_bipartite_projection_matrix`,
  `sdsm`/`fdsm`/`fixed*`, `fastball`, `bicm`), so Family A is essentially built.

## 5. Key finding: Family A already works today

```python
import networkx as nx
import networkx_backbone as nb

hyperedges = {"H1": [1, 2, 3], "H2": [1, 2, 3], "H3": [3, 4, 5, 6], "H4": [5, 6]}
nodes = sorted({v for members in hyperedges.values() for v in members})

B = nx.Graph()                                 # incidence bipartite graph
B.add_nodes_from(nodes, bipartite=0)           # nodes
B.add_nodes_from(hyperedges, bipartite=1)      # hyperedges as "artifacts"
for h, members in hyperedges.items():
    for v in members:
        B.add_edge(v, h)

scored   = nb.sdsm(B, agent_nodes=nodes, projection="hyper")
backbone = nb.threshold_filter(scored, "sdsm_pvalue", 0.30, mode="below")
# backbone.edges() -> [(1, 2), (4, 5), (4, 6), (5, 6)]   (verified)
```

A large slice of "hypergraph backboning" is therefore a **latent, undocumented
capability** today (projection family). The paper's method is a *different* output
type (a sub-hypergraph) and is the new work.

## 6. Method inventory: generalizations vs. sui generis

### 6.1 Generalizations (reachable once a hypergraph enters the pipeline)

| Hypergraph method | Reduces to | Notes |
|-------------------|-----------|-------|
| Hyperedge global-weight threshold | `global_threshold_filter` | trivial |
| Degree-preserving projection null (SDSM/FDSM/fixed*) | `sdsm`/`fdsm`/`fixed*` on incidence | **already works** (§5) |
| Disparity / MLF / LANS / NC / ECM on the projection | matching statistical filter | applied to the projected graph |
| Clique- or line-graph expansion + any graph backbone | existing graph methods | transform, then any current method |

### 6.2 Sui generis methods (no clean dyadic analog)

| Method | Paradigm | Reference / impl | Distinctness |
|--------|----------|------------------|--------------|
| **MDL hypergraph backbone (this paper)** | B2 (compression) | Kirkley+ 2026 | **Headline.** Global MDL optimum over parent/child overlap & nestedness; parameter-free; weighted via γ; sub-hypergraph output. Naive toplex/maximal-face reduction is a degenerate special case. |
| Statistically Validated Hypergraph / Cores (SVH/SVC) | B1 (statistical) | Musciotto+ 2021; HGX `get_svh`/`get_svc` | Complementary α-based alternative; group-level null-model test. |
| Toplex / inclusion (encapsulation) reduction | structural | HNX `toplexes()`, XGI `encapsulation_dag` | Cheap heuristic; subsumed by the MDL objective. |
| s-connectivity / s-line-graph backbone | structural | HNX `s_components` | Parameterized by shared-node threshold `s`; no dyadic analog. |
| Order-resolved hyperedge filtering | utility | building block | Meaningful only with variable arity. |

**Bottom line for the user's question:** hypergraph backboning is *mostly*
generalizations of existing methods (§6.1), **but the paper's MDL method is
genuinely new** and cannot be produced by the current ensemble — both because the
objective exploits higher-order overlap/nestedness and because the output is a
sub-hypergraph. It is worth porting; SVH/SVC and the structural primitives are
worthwhile but secondary.

## 7. Optional interoperability with hypergraph libraries

All four target libraries converge on an incidence/bipartite substrate and **all
support the HIF JSON interchange format**, so interop is cheap and adds **no
required dependency**.

| Library | Hypergraph type(s) | → incidence / bipartite (in) | ← construct (from our output) | HIF |
|---------|--------------------|------------------------------|-------------------------------|-----|
| **XGI** (`xgi`) | `Hypergraph`, `DiHypergraph`, `SimplicialComplex` | `xgi.to_bipartite_graph`, `xgi.to_incidence_matrix` | `xgi.from_bipartite_graph`, `xgi.from_incidence_matrix` | `xgi.read_hif`/`write_hif` |
| **HyperNetX** (`hypernetx`) | `Hypergraph` | `.bipartite()`, `.incidence_matrix()`, `.incidence_dict` | `Hypergraph.from_bipartite`, `.restrict_to_edges(keep)` | supported |
| **HypergraphX** (`hypergraphx`) | `Hypergraph`, Temporal/Directed/Multiplex | `.binary_incidence_matrix(return_mapping=True)` | `Hypergraph(edge_list=...)` | supported |
| **HAT** (`HAT`) | `Hypergraph` (tensor/incidence) | `.incidence_matrix` | `Hypergraph(incidence_matrix=...)` | import/export |

- **`restrict_to_edges`** (HNX) is the natural way to return the MDL backbone as a
  native object in each library; HGX/HNX also give SVH/toplex references.
- Design: (1) **HIF** as a stdlib-only hub (`read_hif`/`write_hif` into our
  hyperedge-list form); (2) thin lazy `from_*`/`to_*` adapters (~10–20 lines each,
  since the converters above already exist); (3) optional extras
  `networkx-backbone[xgi|hypernetx|hypergraphx|hat]`. Core install unchanged; each
  adapter imports its library lazily and errors with an install hint if absent.

## 8. Gap analysis and challenges

1. **No native hypergraph type in NetworkX.** Use a hyperedge-list / incidence
   representation internally (§9.1); the MDL optimizer's `Int(G)` is itself a
   NetworkX graph.
2. **Sub-hypergraph output.** Doesn't flow through `threshold_filter`. Mirror the
   boolean-flag idiom: annotate hyperedges with an `mdl_keep` role + a
   `hyperedge_filter`, and optionally return a native library object.
3. **Global optimizer, not a per-edge score.** The MDL backbone is a combinatorial
   optimum (greedy), unlike independent per-edge p-values — document it as a
   "direct" method like the spanning-tree/metric backbones.
4. **One knob for weights (γ).** Parameter-free unweighted; `γ` only for weighted,
   default `γ=1`. Far less parameter burden than α-based methods.
5. **Dependency policy preserved.** `networkx` + `numpy`/`scipy`
   (`scipy.special.gammaln`); the four libraries stay optional; HIF is stdlib-only.

## 9. Proposed design and re-scoped plan

### 9.1 Representation

Internal: a **hyperedge list** (tuples/`frozenset`s) + node ordering, with
incidence matrix / incidence bipartite graph available on demand. Zero new deps;
matches the substrate every target library and HIF share.

### 9.2 Phase 0 — ingestion + surface Family A (**implemented**)

`networkx_backbone.hypergraph_io` provides `hypergraph_to_bipartite` (enabling the
existing SDSM/FDSM/fixed projection backbones on hypergraphs), `read_hif`/`write_hif`
(stdlib-only HIF interchange, cross-checked against xgi in tests), and lazy
`from_*`/`to_*` adapters for xgi, HyperNetX, HypergraphX, and HAT. No required
dependency is added. (A worked tutorial is still a nice-to-have follow-up.)

### 9.3 Phase 1 — MDL hypergraph backbone (the paper)

A `hypergraph` module implementing Kirkley+ 2026:

```python
# networkx_backbone/hypergraph.py
def mdl_hypergraph_backbone(
    hyperedges, weights=None, gamma=1.0, prior="poisson",
    method="auto",          # "node" | "edge" | "auto" (run both, keep lower L)
    sample_pairs=None, seed=None,
):
    """Return the MDL-optimal backbone (Kirkley et al. 2026).

    Returns a result with: backbone hyperedges, parent->children assignment,
    description_length, and compression_ratio eta. Also annotates each hyperedge
    with an `mdl_keep` boolean role.
    """

def hyperedge_filter(scored, role="mdl_keep"):
    """Boolean-flag filter -> the backbone sub-hypergraph."""

def hypergraph_compression_ratio(hyperedges, backbone=None, weights=None, gamma=1.0):
    """Inverse compression ratio eta (Eq. 8) as an evaluation measure."""
```

Building blocks: `intersection_graph(hyperedges)` (NetworkX), the parent/child MDL
terms (`gammaln`-based), greedy "node"/"edge" optimizers over the star partition,
and optional pair-sampling for very large/dense inputs. The local variant
(Appendix E) is a natural follow-up.

### 9.4 Phase 2 — complementary methods

Structural sui generis methods (**implemented**): `maximal_hyperedges`
(inclusion/toplex reduction), `order_filter` (order-resolved filtering), and
`intersection_graph(..., s=...)` + `s_components` (s-line graph / s-connectivity).
These are stdlib + networkx only and return plain hyperedge collections.

Statistical sui generis methods (**implemented**): `statistically_validated_hypergraph`
(SVH) and `statistically_validated_cores` (SVC) — the Musciotto, Battiston &
Mantegna (2021) null-model validation, faithfully re-implemented from HGX
`get_svh`/`get_svc` with a lazy `scipy` dependency (binomial tail + Benjamini-Hochberg
FDR). P-values verified exactly against `scipy.stats.binom`.

### 9.5 Testing strategy

- Phase 0: round-trip converter tests per library + HIF; equivalence of
  `hypergraph_to_bipartite + sdsm` with the manual recipe.
- Phase 1: **reproduce the paper's controlled synthetic experiments** — planted
  fully-nested simplices and parent/child-with-noise hypergraphs (Figs. 1–3):
  backbone recovers planted top faces; `η` behaves as reported; greedy ≈ exact on
  tiny inputs; `γ=1` ≡ unweighted; `γ→0` forces high-weight edges in; determinism
  via `seed`.
- Cross-validate against HGX/HNX where installed (skipped otherwise).

## 10. Resolved questions and remaining implementation decisions

The first-draft "open questions" are now answered by the paper:
family = **B2 (MDL compression)**; null model = **none** (information-theoretic,
parameter-free unweighted); correction = **n/a**; output = **sub-hypergraph** + star
forest + `η`; relationship to SVH/SVC = **distinct paradigm**.

Resolved during implementation:
1. Optimizer — **both** greedy schemes implemented (`method="edge"`, `"node"`),
   with `method="auto"` (the default) running both and keeping the lower
   description length, per the paper.
2. Weight prior default `poisson`, `γ` default `1.0`.
3. Result object — a `HypergraphBackbone` dataclass (backbone, assignment, `η`,
   description lengths).

Remaining nice-to-have: the Appendix E local (per-node-neighbourhood) variant.

## 11. References

- **Kirkley, Felippe, Malizia & Battiston — Hypergraph backboning — arXiv:2606.00893 (2026).**
- Musciotto, Battiston & Mantegna — Detecting informative higher-order interactions
  in statistically validated hypergraphs — Communications Physics (2021),
  arXiv:2103.16484. https://www.nature.com/articles/s42005-021-00710-4
- Backbone 3.0: An R package for extracting network backbones — PLOS One (2026).
  https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0349258
- Coscia & Neffke — Network backboning with noisy data — arXiv:1906.09081.
- HIF: The hypergraph interchange format — arXiv:2507.11520;
  standard: https://github.com/HIF-org/HIF-standard
- Hypergraphx (HGX) — arXiv:2303.15356. https://github.com/HGX-Team/hypergraphx
- HyperNetX (HNX) — arXiv:2310.11626. https://github.com/pnnl/HyperNetX
- XGI — https://github.com/xgi-org/xgi
- Hypergraph Analysis Toolbox (HAT) — https://github.com/Jpickard1/Hypergraph-Analysis-Toolbox
