# Coverage evaluation: Neal's *Backbone 3.0* (PLOS ONE, 2026)

- **Status:** Evaluation + gap proposals (no library code changed by this document)
- **Tracking:** open items now live in `docs/roadmap.rst` and GitHub issue #31
- **Trigger:** Evaluate `networkx-backbone`'s coverage of the methods in
  *Neal, Z. P. (2026), "Backbone 3.0: An R package for extracting network
  backbones," PLOS ONE* (DOI 10.1371/journal.pone.0349258), propose
  implementations for any gaps, and add the paper to the references.

> Note on sources: the PLOS article and PDF are not reachable from this
> environment's network allowlist. This evaluation is based on the authoritative
> source of the R package itself — `zpneal/backbone` **v3.0.4** (DESCRIPTION,
> NAMESPACE, `NEWS.md`, and the `R/backbone_from_*` sources, fetched from
> `raw.githubusercontent.com`) — cross-checked with the paper's metadata.

---

## 1. Summary

**Every backbone *model* in Backbone 3.0 is already implemented in
`networkx-backbone`, including its hypergraph-projection capability.** The
package's exported models map one-to-one onto functions here, and this library
goes well beyond Backbone 3.0 (noise-corrected, ECM, multiscale linkage,
structural, proximity, and a full hypergraph module: MDL backbone, SVH/SVC,
interop).

The differences are **features layered on the statistical models**, not missing
models. Backbone 3.0 adds four cross-cutting options; three are now implemented
in `networkx-backbone` and the fourth is parked:

| Feature | What it is | Status |
|---------|-----------|--------|
| **`mtc`** multiple-testing correction | Bonferroni / Holm / Hochberg / BH (fdr) / BY adjustment of edge p-values before thresholding | ✅ `adjust_pvalues`, `threshold_filter(mtc=...)` |
| **`signed`** backbones | Two-tailed test retaining significantly *strong* (+) and significantly *weak* (−) edges, with a `sign` attribute | ✅ `signed=` on disparity/mlf/lans + sdsm/fdsm/fixed* |
| **SDSM-EC** edge constraints | `sdsm` with prohibited/required edges (structural 0s/1s; Neal & Neal 2023) | ✅ `sdsm(prohibited=, required=)` |
| **`narrative`** | Auto-generated methods text + citations for a chosen backbone | Parked (low value; utility only) |

## 2. What Backbone 3.0 provides

Backbone 3.0 is organized by **input type**, with a model chosen per input
(mirroring this library's own `backbone_from_weighted` / `_from_projection` /
`_from_unweighted` wrappers):

- **Weighted networks** (`backbone_from_weighted`): models `disparity`, `lans`,
  `mlf`, and `global` (with a length-2 parameter giving a *signed* global
  threshold). Controlled by `alpha`, `signed`, `mtc`, `missing_as_zero`.
- **Bipartite projections *and hypergraphs*** (`backbone_from_projection`): the
  input `B` is a bipartite network **or a hypergraph, as an incidence matrix**;
  models `sdsm` (incl. **SDSM-EC** when structural 0s/1s are present), `fdsm`,
  `fixedfill`, `fixedrow`, `fixedcol`. Controlled by `alpha`, `signed`, `mtc`,
  `missing_as_zero`, `trials`.
- **Unweighted networks** (`backbone_from_unweighted`): `sparsify`-family.
- A unified `backbone()` wrapper dispatching on input class, plus `bicm`,
  `fastball`, and `print`/`summary`/`plot` for backbone objects.

New since 2.x (from `NEWS.md`): modular rewrite organized by input type; the
`backbone()` wrapper; **hypergraph** projection input; SDSM-EC structural
constraints; backbone objects with `narrative`; removal of edgelist input and the
ordinal SDSM.

## 3. Coverage matrix (models)

| Backbone 3.0 export / model | `networkx-backbone` equivalent | Status |
|-----------------------------|-------------------------------|--------|
| `disparity` | `disparity_filter` / `disparity` | ✅ |
| `lans` | `lans_filter` / `lans` | ✅ |
| `mlf` | `marginal_likelihood_filter` / `mlf` | ✅ |
| `global` (threshold) | `global_threshold_filter` | ✅ (one-sided; signed variant → §4.2) |
| `sdsm` | `sdsm` | ✅ (edge constraints → §4.3) |
| `fdsm` | `fdsm` | ✅ |
| `fixedfill` / `fixedrow` / `fixedcol` | `fixedfill` / `fixedrow` / `fixedcol` | ✅ |
| `bicm` | `bicm` | ✅ |
| `fastball` | `fastball` | ✅ |
| `sparsify` (unweighted) | `sparsify` / `lspar` / `local_degree` | ✅ |
| `backbone_from_weighted/_projection/_unweighted`, `backbone` | same names | ✅ |
| **hypergraph** projection input | `hypergraph_to_bipartite` → `sdsm`/`fdsm`/`fixed*` | ✅ |

**Beyond Backbone 3.0:** `noise_corrected_filter`, `ecm_filter`,
`multiple_linkage_analysis`; the entire `structural` and `proximity` families;
and the `hypergraph` module (`mdl_hypergraph_backbone`,
`statistically_validated_hypergraph`/`_cores`, `maximal_hyperedges`,
`order_filter`, `s_components`) with HIF/xgi/HyperNetX/HypergraphX/HAT interop.

## 4. Gap analysis and proposed implementations

### 4.1 Multiple-testing correction (`mtc`) — ✅ implemented

Backbone 3.0 adjusts the matrix of edge p-values with R's `p.adjust()` before
thresholding at `alpha`. `networkx-backbone` now provides `adjust_pvalues()`
(reproducing R's `p.adjust` for `bonferroni`, `holm`, `hochberg`, `bh`/`fdr`,
`by`; verified against `scipy.stats.false_discovery_control`) and an `mtc`
parameter on `threshold_filter`, so every p-value method is FDR/Bonferroni-aware
through one shared path. Original API sketch:

```python
# networkx_backbone/filters.py
def adjust_pvalues(pvalues, method="bh"):
    """Return multiplicity-adjusted p-values.

    method in {"bonferroni","holm","hochberg","hommel","bh"/"fdr","by"}.
    """

def threshold_filter(G, score, threshold, mode="below", *, mtc="none", ...):
    """When mtc != "none" and mode == "below", adjust the score attribute across
    all edges with adjust_pvalues(..., mtc) before applying the threshold."""
```

`adjust_pvalues` is ~30 lines of pure Python (Bonferroni/Holm step-down,
Benjamini–Hochberg/Yekutieli step-up); the existing `_bh_threshold` in
`hypergraph.py` is the BH building block. This makes every p-value method
(`disparity`, `mlf`, `lans`, `sdsm`, `fdsm`, `fixed*`) FDR/Bonferroni-aware
through one shared path, matching Backbone 3.0's `mtc` semantics.

### 4.2 Signed backbones (`signed`) — ✅ implemented

With `signed=TRUE`, Backbone 3.0 runs a **two-tailed** test and keeps edges that
are significantly *strong* (sign `+1`) **and** significantly *weak* (sign `−1`),
annotating each retained edge with a `sign`.

`networkx-backbone` now exposes `signed=False` on the weighted scorers
(`disparity_filter`/`mlf`/`lans_filter`) and the projection null models
(`sdsm`/`fdsm`/`fixedfill`/`fixedrow`/`fixedcol`). Each adds the lower-tail
p-value, stores a two-sided p-value (`2·min(p_hi, p_lo)`, clipped to 1) and a
`"sign"` edge attribute (`+1` strong, `−1` weak). The `sign` flows through
`threshold_filter`/`boolean_filter` automatically (edge data is copied).
`backbone_from_weighted` and `backbone_from_projection` forward `signed`.

### 4.3 SDSM with edge constraints (SDSM-EC) — ✅ implemented

Backbone 3.0's `sdsm` switches to **SDSM-EC** (Neal & Neal 2023) when the
incidence matrix carries structural values: `10` = prohibited edge, `11` =
required edge.

`networkx-backbone`'s `sdsm` now accepts `prohibited`/`required` as iterables of
`(agent, artifact)` pairs and fixes those cells' null probability to 0/1 before
the Poisson-binomial test. Defaults (`None`) leave behavior unchanged; unknown
nodes are ignored; constraints compose with `signed` and are forwarded by
`backbone_from_projection`.

### 4.4 Narrative (`narrative`) — parked

Backbone 3.0 can emit suggested methods text and citations for a chosen backbone.

**Proposal (low priority).** A `describe_backbone(method, alpha=..., mtc=...)`
helper returning a citation/methods string per method. Pure formatting; no
algorithmic content. Could also populate a `narrative` field on the hypergraph
result objects.

### 4.5 Non-gaps

- **Hypergraph input** — covered (§3); this library additionally offers native
  hypergraph backbones beyond projection.
- **`missing_as_zero`** — the projection scorers already test *all* agent pairs
  (including zero co-occurrence), so absent edges are evaluated; a weighted-input
  `missing_as_zero` toggle is a minor convenience if desired.
- **`print`/`summary`/`plot`** — this library returns NetworkX graphs and provides
  its own `visualization` module and `measures` (`compare_backbones`).

## 5. Status

**`mtc`, `signed`, and SDSM-EC are implemented** (additive parameters; no change
to default behavior), closing the methodological gaps versus Backbone 3.0. Only
`narrative` (auto methods text — a utility, not an algorithm) is parked for a
future follow-up.

## 6. References

- Neal, Z. P. (2026). *Backbone 3.0: An R package for extracting network
  backbones.* PLOS ONE. https://doi.org/10.1371/journal.pone.0349258
- Neal, Z. P. (2022). *backbone: An R package to extract network backbones.* PLOS
  ONE, 17(5), e0269137. https://doi.org/10.1371/journal.pone.0269137
- Neal, Z. P., & Neal, J. W. (2023). *Stochastic Degree Sequence Model with Edge
  Constraints (SDSM-EC) for Backbone Extraction.* Complex Networks 12, 127-136.
  https://doi.org/10.1007/978-3-031-53468-3_11
