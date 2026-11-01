"""
Hypergraph backbone methods.

These methods operate directly on hypergraphs (collections of hyperedges, where
each hyperedge is a set of nodes of arbitrary size) rather than on dyadic graphs.

A hyperedge is given as any iterable of node labels; a hypergraph is an iterable
of hyperedges, for example ``[(1, 2, 3), (2, 4), (1, 2, 3, 5)]``.  NetworkX has
no native hypergraph type, so inputs and outputs use plain Python collections
(the backbone is returned as a list of :class:`frozenset` hyperedges).

Methods
-------
mdl_hypergraph_backbone
    Parameter-free information-theoretic (MDL) backbone that prunes nested and
    redundant hyperedges (Kirkley, Felippe, Malizia & Battiston, 2026).
hypergraph_compression_ratio
    Inverse compression ratio ``eta`` achieved by the MDL backbone.
intersection_graph
    Graph linking hyperedges that share at least one node.
"""

import math
from dataclasses import dataclass, field

import networkx as nx

from networkx_backbone._docstrings import append_complexity_docstrings

__all__ = [
    "intersection_graph",
    "mdl_hypergraph_backbone",
    "hypergraph_compression_ratio",
    "HypergraphBackbone",
    "maximal_hyperedges",
    "order_filter",
    "s_components",
    "statistically_validated_hypergraph",
    "statistically_validated_cores",
    "ValidatedHypergraph",
    "svh",
    "svc",
]

_LN2 = math.log(2.0)


# ---------------------------------------------------------------------------
# Information-theoretic primitives (all codelengths in bits, log base 2)
# ---------------------------------------------------------------------------


def _log2(x):
    return math.log2(x)


def _log2_factorial(n):
    """log2(n!) via the log-gamma function (stdlib, no numpy/scipy needed)."""
    return math.lgamma(n + 1.0) / _LN2


def _log2_binom(n, k):
    """log2 of the binomial coefficient C(n, k)."""
    if k < 0 or k > n or n < 0:
        return float("-inf")
    return (
        math.lgamma(n + 1.0) - math.lgamma(k + 1.0) - math.lgamma(n - k + 1.0)
    ) / _LN2


def _parent_codelength(size, n_orders, n_nodes):
    """H(p): bits to transmit a parent (backbone) hyperedge.  Eq. (1)."""
    return _log2(n_orders) + _log2_binom(n_nodes, size)


def _child_codelength(size_c, size_p, overlap, n_orders, n_nodes):
    """H(c|p): bits to transmit a child hyperedge from its parent.  Eq. (3)."""
    return (
        _log2(n_orders)
        + _log2(min(size_p, size_c))
        + _log2_binom(size_p, overlap)
        + _log2_binom(n_nodes - size_p, size_c - overlap)
    )


def _reduced_mutual_information(size_p, size_c, overlap, n_nodes):
    """R(c, p): reduced mutual information between hyperedges.  Eq. (11).

    Symmetric in the two hyperedge sizes.  Larger overlap / nestedness gives a
    larger value, so the description length favours assigning highly overlapping
    hyperedges as children of a shared parent.
    """
    p_not_c = size_p - overlap
    c_not_p = size_c - overlap
    rest = n_nodes - size_p - c_not_p  # = n_nodes - |p ∪ c|
    log2_multinomial = (
        math.lgamma(n_nodes + 1.0)
        - math.lgamma(overlap + 1.0)
        - math.lgamma(p_not_c + 1.0)
        - math.lgamma(c_not_p + 1.0)
        - math.lgamma(rest + 1.0)
    ) / _LN2
    return (
        _log2_binom(n_nodes, size_p)
        + _log2_binom(n_nodes, size_c)
        - log2_multinomial
        - _log2(min(size_p, size_c))
    )


# ---------------------------------------------------------------------------
# Weight model (empirical-Bayes Poisson / Geometric prior).  Sec. III.
# ---------------------------------------------------------------------------


def _expected_weight(role, gamma, mean_weight):
    """Expected weight mu_b for a parent (role=1) or child (role=0).  Eq. (15)."""
    return 1.0 + 2.0 * (gamma ** (1 - role)) * (mean_weight - 1.0) / (1.0 + gamma)


def _weight_codelength(weight, role, gamma, mean_weight, prior):
    """L(w, b): bits to transmit a hyperedge weight given its role.  Eqs (16)-(17)."""
    mu = _expected_weight(role, gamma, mean_weight)
    if prior == "poisson":
        if mu - 1.0 <= 0.0:
            return 0.0
        return (
            (mu - 1.0) / _LN2
            - (weight - 1.0) * _log2(mu - 1.0)
            + _log2_factorial(weight - 1.0)
        )
    if prior == "geometric":
        if mu <= 1.0:
            return 0.0
        return _log2(mu) + (weight - 1.0) * _log2(mu / (mu - 1.0))
    raise ValueError(f"prior must be 'poisson' or 'geometric', got {prior!r}")


def _weight_term(weight, gamma, mean_weight, prior):
    """L(w, 1) - L(w, 0): weight-dependent reward for keeping an edge as a parent.

    Negative for high-weight hyperedges when ``gamma < 1``, which discourages
    demoting them to children (i.e. encourages keeping them in the backbone).
    Zero when ``gamma == 1`` or the hypergraph is effectively unweighted.
    """
    return _weight_codelength(
        weight, 1, gamma, mean_weight, prior
    ) - _weight_codelength(weight, 0, gamma, mean_weight, prior)


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------


def _normalize_hyperedges(hyperedges, weights):
    """Validate input and collapse duplicate / empty hyperedges.

    Returns ``(edges, weights)`` where ``edges`` is a list of unique
    :class:`frozenset` hyperedges and ``weights`` a parallel list of floats.
    Duplicate hyperedges are merged; their weights are summed.
    """
    raw = [frozenset(e) for e in hyperedges]
    if weights is not None:
        weights = [float(w) for w in weights]
        if len(weights) != len(raw):
            raise ValueError("weights must have the same length as hyperedges")
        if any(w < 1.0 for w in weights):
            raise ValueError("weights must be >= 1")

    edges = []
    out_weights = []
    index = {}
    for i, fs in enumerate(raw):
        if len(fs) == 0:
            continue
        w = weights[i] if weights is not None else 1.0
        if fs in index:
            if weights is not None:
                out_weights[index[fs]] += w
        else:
            index[fs] = len(edges)
            edges.append(fs)
            out_weights.append(w)
    return edges, out_weights


def _sorted_labels(labels):
    """Sort node labels deterministically, tolerating mixed / unorderable types.

    Uses the labels' natural order when they are mutually comparable (so
    integer labels sort numerically) and falls back to ``repr`` otherwise.
    """
    labels = list(labels)
    try:
        return sorted(labels)
    except TypeError:
        return sorted(labels, key=repr)


def _sorted_by(items, key):
    """``sorted(items, key=key)``, retrying on ``repr``-ed labels on TypeError.

    *key* is called as ``key(item, sort_labels)`` where ``sort_labels`` is a
    function returning a canonical sorted list of an iterable of node labels.
    """
    items = list(items)
    try:
        return sorted(items, key=lambda it: key(it, sorted))
    except TypeError:
        return sorted(items, key=lambda it: key(it, lambda xs: sorted(map(repr, xs))))


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class HypergraphBackbone:
    """Result of :func:`mdl_hypergraph_backbone`.

    Attributes
    ----------
    backbone : list of frozenset
        The retained ("parent") hyperedges forming the structural backbone.
    assignment : dict
        Mapping from each parent hyperedge to the list of child hyperedges it
        encodes (its star in the intersection graph).  Parents with no children
        are absent from this mapping.
    n_nodes : int
        Number of distinct nodes in the input hypergraph.
    n_orders : int
        Number of distinct hyperedge sizes (orders) in the input.
    weighted : bool
        Whether edge weights influenced the backbone.
    gamma : float
        Weight/topology trade-off used (only meaningful when ``weighted``).
    prior : str
        Weight prior used (``"poisson"`` or ``"geometric"``).
    description_length : float
        Description length ``L(G, B*)`` of the input given the backbone (bits).
    baseline_description_length : float
        Description length ``L(G, G)`` with no backbone (bits).
    compression_ratio : float
        Inverse compression ratio ``eta = L(G, B*) / L(G, G)`` in ``[0, 1]``.
        Smaller means more redundant structure was removed.
    """

    backbone: list = field(default_factory=list)
    assignment: dict = field(default_factory=dict)
    n_nodes: int = 0
    n_orders: int = 0
    weighted: bool = False
    gamma: float = 1.0
    prior: str = "poisson"
    description_length: float = 0.0
    baseline_description_length: float = 0.0
    compression_ratio: float = 1.0

    def __len__(self):
        return len(self.backbone)

    @property
    def n_input_hyperedges(self):
        """Total hyperedges = backbone parents + all children."""
        return len(self.backbone) + sum(len(v) for v in self.assignment.values())

    @property
    def fraction_kept(self):
        """Fraction of (unique) input hyperedges retained in the backbone."""
        total = self.n_input_hyperedges
        return len(self.backbone) / total if total else 1.0


@dataclass
class ValidatedHypergraph:
    """Result of :func:`statistically_validated_hypergraph` / cores.

    Attributes
    ----------
    validated : list of frozenset
        The statistically validated hyperedges (SVH) or cores (SVC) -- the
        sub-hypergraph that survived FDR validation.
    pvalues : dict
        Mapping from each tested hyperedge/group to its p-value.
    counts : dict
        Mapping from each tested hyperedge/group to its observed co-occurrence
        count (multiplicity for SVH; number of containing instances for SVC).
    alpha : float
        Significance level used for FDR validation.
    method : str
        ``"svh"`` or ``"svc"``.
    n_nodes : int
        Number of distinct nodes in the input hypergraph.
    n_instances : int
        Total number of hyperedge instances (sum of multiplicities).
    """

    validated: list = field(default_factory=list)
    pvalues: dict = field(default_factory=dict)
    counts: dict = field(default_factory=dict)
    alpha: float = 0.01
    method: str = "svh"
    n_nodes: int = 0
    n_instances: int = 0

    def __len__(self):
        return len(self.validated)

    def __iter__(self):
        return iter(self.validated)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def intersection_graph(hyperedges, s=1, weight="overlap"):
    """Build the intersection graph (or s-line graph) of a hypergraph.

    Each hyperedge becomes a node (labelled by its integer index after
    deduplication); two hyperedges are linked when they share at least *s*
    nodes.  With ``s=1`` this is the intersection graph over which MDL
    parent--child relationships are formed (Kirkley et al. 2026, Appendix D);
    with ``s > 1`` it is the *s-line graph*, the basis of s-connectivity in
    higher-order networks.

    Parameters
    ----------
    hyperedges : iterable of iterables
        The hypergraph.  Each hyperedge is an iterable of node labels.
    s : int, optional (default=1)
        Minimum shared-node count ``|e_i ∩ e_j|`` for two hyperedges to be
        linked.  Must be ``>= 1``.
    weight : string, optional (default="overlap")
        Edge attribute name used to store the overlap size ``|e_i ∩ e_j|``.

    Returns
    -------
    I : networkx.Graph
        Graph on hyperedge indices.  Each node has a ``"members"`` attribute
        (the hyperedge as a :class:`frozenset`); each edge stores the overlap
        size under *weight*.

    Raises
    ------
    ValueError
        If *s* is less than 1.

    See Also
    --------
    s_components : Connected components of the s-line graph.
    hypergraph_to_bipartite : Node--hyperedge incidence graph.

    Examples
    --------
    >>> from networkx_backbone import intersection_graph
    >>> I = intersection_graph([(1, 2, 3), (2, 3, 4), (5, 6)])
    >>> I.number_of_nodes()
    3
    >>> I[0][1]["overlap"]
    2
    >>> intersection_graph([(1, 2, 3), (2, 3, 4), (5, 6)], s=3).number_of_edges()
    0
    """
    if s < 1:
        raise ValueError(f"s must be >= 1, got {s}")

    edges, _ = _normalize_hyperedges(hyperedges, None)
    graph = nx.Graph()
    for i, e in enumerate(edges):
        graph.add_node(i, members=e)

    node_to_edges = {}
    for i, e in enumerate(edges):
        for v in e:
            node_to_edges.setdefault(v, []).append(i)

    overlaps = {}
    for incident in node_to_edges.values():
        m = len(incident)
        if m < 2:
            continue
        for a in range(m):
            for b in range(a + 1, m):
                i, j = incident[a], incident[b]
                key = (i, j) if i < j else (j, i)
                overlaps[key] = overlaps.get(key, 0) + 1

    for (i, j), o in overlaps.items():
        if o >= s:
            graph.add_edge(i, j, **{weight: o})
    return graph


def _greedy_edge(n_edges, edges, neighbors, wterm):
    """Greedy "edge" optimiser: accept parent->child links by decreasing gain.

    Builds a star partition of the intersection graph by greedily accepting the
    highest-gain parent--child assignment that preserves the star structure
    (Kirkley et al. 2026, Appendix D).  Returns ``parent_of`` (child index ->
    parent index).
    """
    candidates = []
    for i in range(n_edges):
        si = len(edges[i])
        for j, rmi in neighbors[i]:
            gain = rmi + wterm[i]  # i becomes a child of parent j
            if gain > 0.0:
                candidates.append((gain, len(edges[j]), si, i, j))
    # Highest gain first; prefer the larger hyperedge as parent on ties.
    candidates.sort(key=lambda t: (-t[0], -t[1], t[2], t[3], t[4]))

    UNDECIDED, PARENT, CHILD = 0, 1, 2
    role = [UNDECIDED] * n_edges
    parent_of = {}
    for _gain, _psize, _csize, c, p in candidates:
        if role[c] != UNDECIDED or role[p] == CHILD:
            continue
        role[c] = CHILD
        parent_of[c] = p
        if role[p] == UNDECIDED:
            role[p] = PARENT
    return parent_of


def _greedy_node(n_edges, neighbors, wterm):
    """Greedy "node" optimiser: add backbone parents by decreasing total gain.

    The alternative scheme of Kirkley et al. 2026 (Appendix D): repeatedly add to
    the backbone the hyperedge whose promotion to a parent most increases the
    total parent--child savings (a facility-location-style greedy), then force any
    still-uncovered hyperedge to be a parent.  Returns ``parent_of``.

    The marginal gain of promoting a hyperedge never increases as the backbone
    grows (each child's best saving only rises, and backbone members drop out of
    the sum), so the argmax is found with a *lazy* max-heap of stale gains
    (CELF): only the top entry is re-evaluated each step.  Ties are broken by
    the smallest hyperedge index, exactly as an exhaustive scan would, so the
    result is identical to recomputing every gain each round.
    """
    import heapq

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

    def _gain(e):
        gain = -best_saving[e]
        for c, rmi in neighbors[e]:
            if in_backbone[c]:
                continue
            delta = (rmi + wterm[c]) - best_saving[c]
            if delta > 0.0:
                gain += delta
        return gain

    min_gain = 1e-12  # require a strictly positive improvement
    heap = [(-_gain(e), e) for e in range(n_edges)]
    heapq.heapify(heap)
    while heap:
        neg_stale, e = heap[0]
        if -neg_stale <= min_gain:
            break  # every true gain is bounded by its stale value
        fresh = _gain(e)
        if fresh != -neg_stale:
            heapq.heapreplace(heap, (-fresh, e))
            continue
        heapq.heappop(heap)
        in_backbone[e] = True
        best_saving[e] = 0.0
        best_parent[e] = None
        _attach_children(e)

    # Force-cover any hyperedge still without a parent (no overlap with backbone).
    for e in range(n_edges):
        if not in_backbone[e] and best_parent[e] is None:
            in_backbone[e] = True
            _attach_children(e)

    return {e: best_parent[e] for e in range(n_edges) if not in_backbone[e]}


def _description_length(
    edges,
    parent_of,
    weight_list,
    n_nodes,
    n_orders,
    weighted,
    gamma,
    mean_weight,
    prior,
):
    """Return ``(L(G, B), L(G, G))`` in bits for a given parent/child assignment."""
    dl = 0.0
    dl0 = 0.0
    for i, e in enumerate(edges):
        size = len(e)
        dl0 += _parent_codelength(size, n_orders, n_nodes)
        if weighted:
            dl0 += _weight_codelength(weight_list[i], 1, gamma, mean_weight, prior)
        if i in parent_of:
            p = parent_of[i]
            overlap = len(e & edges[p])
            dl += _child_codelength(size, len(edges[p]), overlap, n_orders, n_nodes)
            if weighted:
                dl += _weight_codelength(weight_list[i], 0, gamma, mean_weight, prior)
        else:
            dl += _parent_codelength(size, n_orders, n_nodes)
            if weighted:
                dl += _weight_codelength(weight_list[i], 1, gamma, mean_weight, prior)
    return dl, dl0


def mdl_hypergraph_backbone(
    hyperedges,
    weights=None,
    gamma=1.0,
    prior="poisson",
    method="auto",
):
    """Extract a hypergraph backbone via minimum description length (MDL).

    Implements the parameter-free information-theoretic backboning method of
    Kirkley, Felippe, Malizia & Battiston [1]_.  The backbone ``B`` is a subset
    of hyperedges ("parents") chosen so that the remaining "child" hyperedges can
    be cheaply reconstructed from a parent they overlap, exploiting the nested
    and redundant structure unique to higher-order networks.  The optimal
    backbone minimises the two-part description length
    ``L(G, B) = L(B) + L(G|B)`` (Eqs (1)-(6)); equivalently it maximises the
    total parent--child reduced mutual information (Eq. (12)).

    The method is **fully nonparametric** for unweighted hypergraphs.  Edge
    weights are incorporated through an empirical-Bayes prior with a single knob
    *gamma* trading off weight against topology (Sec. III).

    Parameters
    ----------
    hyperedges : iterable of iterables
        The hypergraph.  Each hyperedge is an iterable of node labels (treated as
        a set; repeated nodes within a hyperedge are ignored).  Duplicate
        hyperedges are merged.
    weights : iterable of numbers or None, optional (default=None)
        Optional per-hyperedge weights (each ``>= 1``), parallel to
        *hyperedges*.  ``None`` means an unweighted hypergraph.
    gamma : float, optional (default=1.0)
        Weight/topology trade-off in ``(0, 1]`` (Eq. (14)).  ``gamma=1`` ignores
        weights (recovers the unweighted backbone); ``gamma -> 0`` makes it
        increasingly costly to leave a high-weight hyperedge out of the backbone.
        Ignored when *weights* is ``None``.
    prior : {"poisson", "geometric"}, optional (default="poisson")
        Weight prior family (Eqs (16)-(17)).  Ignored when *weights* is ``None``.
    method : {"auto", "edge", "node"}, optional (default="auto")
        Greedy optimiser (Kirkley et al. 2026, Appendix D).  ``"edge"`` adds
        parent--child links by decreasing gain; ``"node"`` adds backbone parents
        by decreasing total gain; ``"auto"`` runs both and keeps the lower
        description length (the procedure used in [1]_).

    Returns
    -------
    result : HypergraphBackbone
        The backbone hyperedges, the parent--child assignment, and compression
        diagnostics (see :class:`HypergraphBackbone`).

    Raises
    ------
    ValueError
        If *gamma* is not in ``(0, 1]``, *prior* is unknown, *weights* has the
        wrong length or contains a value ``< 1``, or *method* is unsupported.

    See Also
    --------
    hypergraph_compression_ratio : Only the compression ratio ``eta``.
    maximal_hyperedges : Purely set-inclusion (toplex) pruning.
    statistically_validated_hypergraph : Statistical hyperedge filtering.

    Notes
    -----
    Exact minimisation is combinatorial; this uses the greedy heuristics of
    [1]_, which forms a maximum-reward partition of the intersection graph into
    disjoint stars (each child attached to a single parent).  On small inputs the
    greedy compression matches exhaustive search.

    References
    ----------
    .. [1] Kirkley, A., Felippe, H., Malizia, F., & Battiston, F. (2026).
       Hypergraph backboning. arXiv:2606.00893.

    Examples
    --------
    >>> from networkx_backbone import mdl_hypergraph_backbone
    >>> # A 4-node hyperedge with two nested (redundant) sub-hyperedges.
    >>> G = [(1, 2, 3, 4), (1, 2, 3), (2, 3, 4), (8, 9)]
    >>> result = mdl_hypergraph_backbone(G)
    >>> frozenset({1, 2, 3, 4}) in result.backbone
    True
    >>> result.compression_ratio <= 1.0
    True
    """
    if not 0.0 < gamma <= 1.0:
        raise ValueError(f"gamma must be in (0, 1], got {gamma}")
    if prior not in ("poisson", "geometric"):
        raise ValueError(f"prior must be 'poisson' or 'geometric', got {prior!r}")
    if method not in ("edge", "node", "auto"):
        raise ValueError(f"method must be 'edge', 'node', or 'auto', got {method!r}")

    edges, weight_list = _normalize_hyperedges(hyperedges, weights)
    n_edges = len(edges)

    result = HypergraphBackbone(gamma=gamma, prior=prior)
    if n_edges == 0:
        return result

    all_nodes = set()
    sizes = set()
    for e in edges:
        all_nodes.update(e)
        sizes.add(len(e))
    n_nodes = len(all_nodes)
    n_orders = len(sizes)
    result.n_nodes = n_nodes
    result.n_orders = n_orders

    mean_weight = sum(weight_list) / n_edges
    weighted = weights is not None and any(w != 1.0 for w in weight_list)
    result.weighted = weighted

    # Pairwise overlaps via node neighbourhoods (Appendix D).
    node_to_edges = {}
    for i, e in enumerate(edges):
        for v in e:
            node_to_edges.setdefault(v, []).append(i)
    overlaps = {}
    for incident in node_to_edges.values():
        m = len(incident)
        if m < 2:
            continue
        for a in range(m):
            for b in range(a + 1, m):
                i, j = incident[a], incident[b]
                key = (i, j) if i < j else (j, i)
                overlaps[key] = overlaps.get(key, 0) + 1

    # Precompute per-edge weight terms (0 when unweighted).
    if weighted:
        wterm = [_weight_term(w, gamma, mean_weight, prior) for w in weight_list]
    else:
        wterm = [0.0] * n_edges

    # Intersection-graph adjacency with reduced mutual information per pair.
    neighbors = [[] for _ in range(n_edges)]
    for (i, j), o in overlaps.items():
        rmi = _reduced_mutual_information(len(edges[i]), len(edges[j]), o, n_nodes)
        neighbors[i].append((j, rmi))
        neighbors[j].append((i, rmi))

    if method == "edge":
        parent_of = _greedy_edge(n_edges, edges, neighbors, wterm)
    elif method == "node":
        parent_of = _greedy_node(n_edges, neighbors, wterm)
    else:  # "auto": run both and keep the lower description length.
        parent_of = min(
            (
                _greedy_edge(n_edges, edges, neighbors, wterm),
                _greedy_node(n_edges, neighbors, wterm),
            ),
            key=lambda po: _description_length(
                edges,
                po,
                weight_list,
                n_nodes,
                n_orders,
                weighted,
                gamma,
                mean_weight,
                prior,
            )[0],
        )

    dl, dl0 = _description_length(
        edges,
        parent_of,
        weight_list,
        n_nodes,
        n_orders,
        weighted,
        gamma,
        mean_weight,
        prior,
    )

    result.backbone = [edges[i] for i in range(n_edges) if i not in parent_of]
    assignment = {}
    for c, p in parent_of.items():
        assignment.setdefault(edges[p], []).append(edges[c])
    result.assignment = assignment
    result.description_length = dl
    result.baseline_description_length = dl0
    result.compression_ratio = dl / dl0 if dl0 > 0 else 1.0
    return result


def hypergraph_compression_ratio(hyperedges, weights=None, gamma=1.0, prior="poisson"):
    """Inverse compression ratio ``eta`` of the MDL backbone (Eq. (8)).

    A convenience wrapper returning only
    :attr:`HypergraphBackbone.compression_ratio` from
    :func:`mdl_hypergraph_backbone`.  ``eta`` lies in ``[0, 1]``; values near 0
    indicate a highly redundant (compressible) hypergraph, while ``eta = 1``
    indicates no compressible structure.

    Parameters
    ----------
    hyperedges : iterable of iterables
        The hypergraph.
    weights : iterable of numbers or None, optional (default=None)
        Optional per-hyperedge weights.
    gamma : float, optional (default=1.0)
        Weight/topology trade-off (see :func:`mdl_hypergraph_backbone`).
    prior : {"poisson", "geometric"}, optional (default="poisson")
        Weight prior family.

    Returns
    -------
    eta : float
        Inverse compression ratio in ``[0, 1]``.

    Raises
    ------
    ValueError
        For invalid *weights*, *gamma*, or *prior* (see
        :func:`mdl_hypergraph_backbone`).

    See Also
    --------
    mdl_hypergraph_backbone : The full backbone and diagnostics.

    Examples
    --------
    >>> from networkx_backbone import hypergraph_compression_ratio
    >>> eta = hypergraph_compression_ratio([(1, 2, 3, 4), (1, 2, 3), (2, 3, 4)])
    >>> 0.0 <= eta <= 1.0
    True
    """
    return mdl_hypergraph_backbone(
        hyperedges, weights=weights, gamma=gamma, prior=prior
    ).compression_ratio


def maximal_hyperedges(hyperedges):
    """Inclusion (toplex) reduction: keep only the maximal hyperedges.

    Removes every hyperedge that is a strict subset of another hyperedge,
    retaining the *toplexes* -- hyperedges not contained in any other.  This is
    the simplest structural hypergraph backbone, pruning nested redundancy
    purely by set inclusion (cf. ``HyperNetX``'s ``toplexes``).  For a richer,
    information-theoretic treatment of nested *and* overlapping redundancy, see
    :func:`mdl_hypergraph_backbone`.

    Parameters
    ----------
    hyperedges : iterable of iterables
        The hypergraph.  Duplicate hyperedges are merged.

    Returns
    -------
    maximal : list of frozenset
        The maximal hyperedges, ordered by decreasing size (ties broken by
        sorted node labels; labels of mixed, unorderable types are compared by
        ``repr``).

    See Also
    --------
    mdl_hypergraph_backbone : Information-theoretic pruning of nested *and*
        overlapping redundancy.
    order_filter : Keep hyperedges by size.

    Examples
    --------
    >>> from networkx_backbone import maximal_hyperedges
    >>> sorted(map(sorted, maximal_hyperedges([(1, 2, 3), (1, 2), (2, 3), (4, 5)])))
    [[1, 2, 3], [4, 5]]
    """
    edges, _ = _normalize_hyperedges(hyperedges, None)
    order = _sorted_by(
        range(len(edges)), lambda i, srt: (-len(edges[i]), srt(edges[i]))
    )
    kept = []
    for i in order:
        e = edges[i]
        if not any(e < bigger for bigger in kept):
            kept.append(e)
    return kept


def order_filter(hyperedges, min_order=None, max_order=None, orders=None):
    """Keep hyperedges whose order (size) falls in a range or set.

    Order-resolved filtering has no analog in dyadic graphs, where every edge
    has order 2; in a hypergraph it selects interactions at chosen scales.

    Parameters
    ----------
    hyperedges : iterable of iterables
        The hypergraph.  Duplicate hyperedges are merged.
    min_order : int or None, optional (default=None)
        Keep hyperedges with size ``>= min_order``.
    max_order : int or None, optional (default=None)
        Keep hyperedges with size ``<= max_order``.
    orders : iterable of int or None, optional (default=None)
        If given, keep only hyperedges whose size is in this set (applied in
        addition to *min_order*/*max_order*).

    Returns
    -------
    selected : list of frozenset
        The retained hyperedges, in first-occurrence order.

    Raises
    ------
    ValueError
        If *min_order* and *max_order* are both given and ``min_order > max_order``.

    See Also
    --------
    maximal_hyperedges : Structural (set-inclusion) pruning.
    s_components : Group hyperedges by s-connectivity.

    Examples
    --------
    >>> from networkx_backbone import order_filter
    >>> sorted(map(sorted, order_filter([(1, 2), (1, 2, 3), (1, 2, 3, 4)], min_order=3)))
    [[1, 2, 3], [1, 2, 3, 4]]
    """
    if min_order is not None and max_order is not None and min_order > max_order:
        raise ValueError("min_order must not exceed max_order")
    order_set = set(orders) if orders is not None else None

    edges, _ = _normalize_hyperedges(hyperedges, None)
    selected = []
    for e in edges:
        k = len(e)
        if min_order is not None and k < min_order:
            continue
        if max_order is not None and k > max_order:
            continue
        if order_set is not None and k not in order_set:
            continue
        selected.append(e)
    return selected


def s_components(hyperedges, s=1):
    """Group hyperedges into s-connected components.

    Two hyperedges are *s-adjacent* when they share at least *s* nodes; an
    s-component is a connected component of the resulting s-line graph
    (:func:`intersection_graph` with the same *s*).  s-connectivity is a
    higher-order notion with no dyadic counterpart and underlies s-centrality
    and s-distance analyses of hypergraphs.

    Parameters
    ----------
    hyperedges : iterable of iterables
        The hypergraph.  Duplicate hyperedges are merged.
    s : int, optional (default=1)
        Minimum shared-node count for s-adjacency.  Must be ``>= 1``.

    Returns
    -------
    components : list of list of frozenset
        Each inner list is the hyperedges of one s-connected component,
        ordered by decreasing component size (ties broken by sorted node
        labels; labels of mixed, unorderable types are compared by ``repr``).

    Raises
    ------
    ValueError
        If *s* is less than 1.

    See Also
    --------
    intersection_graph : The s-line graph whose components are returned.
    maximal_hyperedges : Structural (set-inclusion) pruning.

    Examples
    --------
    >>> from networkx_backbone import s_components
    >>> comps = s_components([(1, 2, 3), (2, 3, 4), (5, 6, 7)], s=2)
    >>> [len(c) for c in comps]
    [2, 1]
    """
    edges, _ = _normalize_hyperedges(hyperedges, None)
    graph = intersection_graph(edges, s=s)
    components = [
        [edges[i] for i in sorted(component)]
        for component in nx.connected_components(graph)
    ]
    return _sorted_by(
        components,
        lambda comp, srt: (-len(comp), sorted(srt(e) for e in comp)),
    )


# ---------------------------------------------------------------------------
# Statistically validated hypergraphs (Musciotto, Battiston & Mantegna 2021)
# ---------------------------------------------------------------------------


def _normalize_multiplicities(hyperedges, weights):
    """Collapse duplicate hyperedges into integer multiplicities.

    Unlike :func:`_normalize_hyperedges` (which treats the hypergraph as a set),
    this counts repeated occurrences: with ``weights=None`` a hyperedge appearing
    ``r`` times has multiplicity ``r``; with explicit *weights* the integer
    weights of duplicates are summed.  Multiplicities are interaction counts for
    the statistical filters.
    """
    raw = [frozenset(e) for e in hyperedges]
    if weights is not None:
        weights = list(weights)
        if len(weights) != len(raw):
            raise ValueError("weights must have the same length as hyperedges")

    edges = []
    mult = []
    index = {}
    for i, fs in enumerate(raw):
        if len(fs) == 0:
            continue
        if weights is not None:
            w = weights[i]
            if not (w >= 1 and float(w).is_integer()):
                raise ValueError(
                    "statistical hypergraph filters require integer multiplicities >= 1"
                )
            w = int(w)
        else:
            w = 1
        if fs in index:
            mult[index[fs]] += w
        else:
            index[fs] = len(edges)
            edges.append(fs)
            mult.append(w)
    return edges, mult


def _bh_threshold(pvalues, alpha, n_possible):
    """Benjamini-Hochberg FDR threshold with per-rank increment alpha/n_possible.

    Returns the largest ``i * alpha / n_possible`` such that the i-th smallest
    p-value is below it (0.0 if none), matching Tumminello et al. / HGX.
    """
    n = len(pvalues)
    if n == 0:
        return 0.0
    bonf = alpha / n_possible if n_possible > 0 else alpha
    threshold = 0.0
    for rank, p in enumerate(sorted(pvalues), start=1):
        kv = rank * bonf
        if p < kv:
            threshold = kv
    return threshold


def _svh_pvalue(observed, n_instances, degrees, binom):
    """Upper-tail p-value P(X >= observed) with X ~ Binomial(N, prod d_i / N)."""
    p = 1.0
    for d in degrees:
        p *= d / n_instances
    return float(binom.sf(observed - 1, n_instances, p))


def _svh_pvalues(observed, n_instances, degree_lists, binom):
    """Vectorised :func:`_svh_pvalue` over parallel lists of counts / degrees."""
    import numpy as np

    probs = []
    for degrees in degree_lists:
        p = 1.0
        for d in degrees:
            p *= d / n_instances
        probs.append(p)
    sf = binom.sf(np.asarray(observed) - 1, n_instances, np.asarray(probs))
    return [float(v) for v in np.atleast_1d(sf)]


def _check_alpha(alpha):
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")


def _count_subgroups(sizes, min_order, top, limit):
    """Number of sub-groups of order ``min_order..top`` over hyperedge *sizes*.

    Stops counting (returning a value ``> limit``) as soon as *limit* is passed.
    """
    total = 0
    for size in sizes:
        for k in range(min_order, min(top, size) + 1):
            total += math.comb(size, k)
        if total > limit:
            return total
    return total


def statistically_validated_hypergraph(
    hyperedges, weights=None, max_order=None, alpha=0.01
):
    """Extract the Statistically Validated Hypergraph (SVH).

    Keeps the observed hyperedges that recur (co-occur) significantly more often
    than expected under a null model preserving node activity, following
    Musciotto, Battiston & Mantegna [1]_ (the method implemented as ``get_svh``
    in Hypergraphx).  Each hyperedge of order ``k`` is tested independently per
    order: with ``N`` order-``k`` instances and node activities ``d_i`` (number
    of order-``k`` instances containing node ``i``), the probability of seeing a
    group co-occur at least ``n`` times is ``P(X >= n)`` for
    ``X ~ Binomial(N, prod_i d_i / N)``.  P-values are validated with a
    Benjamini-Hochberg FDR at level *alpha* (corrected for the number of
    possible order-``k`` hyperedges).

    Unlike :func:`mdl_hypergraph_backbone` (an information-theoretic, parameter-
    free method), this is a statistical hypothesis test requiring a significance
    level, and is most informative for **weighted** hypergraphs whose weights are
    integer interaction multiplicities.

    Parameters
    ----------
    hyperedges : iterable of iterables
        The hypergraph.  Duplicate hyperedges are merged (multiplicities summed).
    weights : iterable of int or None, optional (default=None)
        Per-hyperedge integer multiplicities (interaction counts).  ``None``
        treats every hyperedge as occurring once.
    max_order : int or None, optional (default=None)
        Only test hyperedges up to this order (size).  ``None`` tests all orders.
    alpha : float, optional (default=0.01)
        FDR significance level, in ``(0, 1]``.

    Returns
    -------
    result : ValidatedHypergraph
        The validated hyperedges plus per-hyperedge p-values and counts.

    Raises
    ------
    ValueError
        If *alpha* is not in ``(0, 1]``, or *weights* has the wrong length or
        contains a value that is not an integer ``>= 1``.

    See Also
    --------
    statistically_validated_cores : Validates sub-groups (cores) as well.
    mdl_hypergraph_backbone : Parameter-free information-theoretic backbone.

    References
    ----------
    .. [1] Musciotto, F., Battiston, F., & Mantegna, R. N. (2021). Detecting
       informative higher-order interactions in statistically validated
       hypergraphs. *Communications Physics*, 4, 218.

    Examples
    --------
    >>> from networkx_backbone import statistically_validated_hypergraph
    >>> edges = [(1, 2), (1, 2), (1, 2), (1, 3), (2, 4), (5, 6)]
    >>> result = statistically_validated_hypergraph(edges)
    >>> isinstance(result.validated, list)
    True
    """
    from scipy.stats import binom

    _check_alpha(alpha)
    edges, mult = _normalize_multiplicities(hyperedges, weights)
    result = ValidatedHypergraph(alpha=alpha, method="svh")
    if not edges:
        return result

    all_nodes = set()
    for e in edges:
        all_nodes.update(e)
    result.n_nodes = len(all_nodes)
    result.n_instances = sum(mult)

    by_order = {}
    for e, w in zip(edges, mult):
        by_order.setdefault(len(e), []).append((e, w))

    for order, members in by_order.items():
        if order < 2 or (max_order is not None and order > max_order):
            continue
        n_instances = sum(w for _, w in members)
        degree = {}
        order_nodes = set()
        for e, w in members:
            order_nodes.update(e)
            for node in e:
                degree[node] = degree.get(node, 0) + w

        pvals = _svh_pvalues(
            [w for _, w in members],
            n_instances,
            [[degree[n] for n in e] for e, _ in members],
            binom,
        )
        n_possible = math.comb(len(order_nodes), order)
        threshold = _bh_threshold(pvals, alpha, n_possible)

        for (e, w), p in zip(members, pvals):
            result.pvalues[e] = p
            result.counts[e] = w
            if p < threshold:
                result.validated.append(e)

    return result


_SVC_MAX_GROUPS = 1_000_000


def statistically_validated_cores(
    hyperedges,
    weights=None,
    min_order=2,
    max_order=None,
    alpha=0.01,
    max_groups=_SVC_MAX_GROUPS,
):
    """Extract the Statistically Validated Cores (SVC).

    A complement to :func:`statistically_validated_hypergraph` that validates
    significant *groups* (cores) of nodes, including sub-groups that are not
    themselves present as a single hyperedge (the ``get_svc`` method of
    Hypergraphx, built on [1]_).  Orders are processed from high to low; once a
    core is validated, its sub-combinations are not re-tested at lower orders, so
    significance is attributed to the largest validated group.  The co-occurrence
    of a group is the number of hyperedge instances (of any order) containing it,
    tested against ``Binomial(N, prod_i d_i / N)`` with global node activities,
    and validated with the same Benjamini-Hochberg FDR as SVH.

    Parameters
    ----------
    hyperedges : iterable of iterables
        The hypergraph.  Duplicate hyperedges are merged (multiplicities summed).
    weights : iterable of int or None, optional (default=None)
        Per-hyperedge integer multiplicities.  ``None`` treats each as occurring once.
    min_order : int, optional (default=2)
        Smallest group size to test (``>= 1``).
    max_order : int or None, optional (default=None)
        Largest group size to test.  ``None`` uses the largest hyperedge size.
        Every sub-group of every hyperedge up to this size is enumerated -- up to
        ``2**k`` per hyperedge of size ``k`` -- so set it when the hypergraph
        contains large hyperedges.
    alpha : float, optional (default=0.01)
        FDR significance level, in ``(0, 1]``.
    max_groups : int or None, optional (default=1_000_000)
        Safety guard: if the number of candidate sub-groups to enumerate (summed
        over the distinct hyperedges, for orders ``min_order..max_order``)
        exceeds this, a :class:`ValueError` is raised *before* any work is done
        instead of running for an impractically long time.  ``None`` disables
        the guard.

    Returns
    -------
    result : ValidatedHypergraph
        The validated cores plus per-group p-values and co-occurrence counts.

    Raises
    ------
    ValueError
        If *alpha* is not in ``(0, 1]``, *min_order* is less than 1, *weights*
        is invalid, or the enumeration would exceed *max_groups* (lower
        *max_order*, or raise / disable *max_groups*).

    See Also
    --------
    statistically_validated_hypergraph : Validates only observed hyperedges.
    maximal_hyperedges : Purely structural pruning of nested hyperedges.

    References
    ----------
    .. [1] Musciotto, F., Battiston, F., & Mantegna, R. N. (2021). Detecting
       informative higher-order interactions in statistically validated
       hypergraphs. *Communications Physics*, 4, 218.

    Examples
    --------
    >>> from networkx_backbone import statistically_validated_cores
    >>> edges = [(1, 2, 3), (1, 2, 3), (1, 2, 3), (1, 4), (2, 5)]
    >>> result = statistically_validated_cores(edges)
    >>> result.method
    'svc'
    >>> # A single 24-node hyperedge has ~16.7 million sub-groups: bound the order.
    >>> big = [tuple(range(24)), (0, 1), (0, 1), (0, 1)]
    >>> len(statistically_validated_cores(big, max_order=3).pvalues)
    2300
    """
    from itertools import combinations

    from scipy.stats import binom

    _check_alpha(alpha)
    if min_order < 1:
        raise ValueError(f"min_order must be >= 1, got {min_order}")
    edges, mult = _normalize_multiplicities(hyperedges, weights)
    result = ValidatedHypergraph(alpha=alpha, method="svc")
    if not edges:
        return result

    n_instances = sum(mult)
    degree = {}
    all_nodes = set()
    for e, w in zip(edges, mult):
        all_nodes.update(e)
        for node in e:
            degree[node] = degree.get(node, 0) + w
    result.n_nodes = len(all_nodes)
    result.n_instances = n_instances

    largest = max(len(e) for e in edges)
    top = largest if max_order is None else min(max_order, largest)
    if max_groups is not None:
        n_groups = _count_subgroups((len(e) for e in edges), min_order, top, max_groups)
        if n_groups > max_groups:
            raise ValueError(
                f"statistically_validated_cores would enumerate more than "
                f"{max_groups} candidate sub-groups (largest hyperedge has "
                f"{largest} nodes, testing orders {min_order}..{top}); pass a "
                "smaller max_order, or raise max_groups (None disables this "
                "guard)"
            )

    validated_groups = []
    for order in range(top, min_order - 1, -1):
        drop = set()
        for g in validated_groups:
            if len(g) > order:
                drop.update(frozenset(c) for c in combinations(tuple(g), order))

        counts = {}
        for e, w in zip(edges, mult):
            if len(e) >= order:
                for c in combinations(tuple(e), order):
                    fs = frozenset(c)
                    if fs not in drop:
                        counts[fs] = counts.get(fs, 0) + w
        if not counts:
            continue

        groups = list(counts)
        pvals = _svh_pvalues(
            [counts[g] for g in groups],
            n_instances,
            [[degree[n] for n in g] for g in groups],
            binom,
        )
        n_possible = math.comb(len(all_nodes), order)
        threshold = _bh_threshold(pvals, alpha, n_possible)

        for g, p in zip(groups, pvals):
            result.pvalues[g] = p
            result.counts[g] = counts[g]
            if p < threshold:
                result.validated.append(g)
                validated_groups.append(g)

    return result


# Short aliases
svh = statistically_validated_hypergraph
svc = statistically_validated_cores


_COMPLEXITY = {
    "intersection_graph": {
        "time": "O(sum_i |G_i|^2)",
        "space": "O(m + P)",
        "notes": "G_i=hyperedges incident to node i, m=hyperedges, P=overlapping pairs.",
    },
    "mdl_hypergraph_backbone": {
        "time": "O(sum_i |G_i|^2 + P log P)",
        "space": "O(m + P)",
        "notes": (
            "P=overlapping pairs. method='edge' sorts candidates; method='node'/"
            "'auto' run a facility-location greedy up to O(m*(m+P))."
        ),
    },
    "hypergraph_compression_ratio": {
        "time": "O(sum_i |G_i|^2 + P log P)",
        "space": "O(m + P)",
    },
    "maximal_hyperedges": {
        "time": "O(m^2 * k)",
        "space": "O(m)",
        "notes": "m=hyperedges, k=mean hyperedge size.",
    },
    "order_filter": {
        "time": "O(m)",
        "space": "O(m)",
    },
    "s_components": {
        "time": "O(sum_i |G_i|^2)",
        "space": "O(m + P)",
        "notes": "P=overlapping pairs in the s-line graph.",
    },
    "statistically_validated_hypergraph": {
        "time": "O(sum_e |e| + m log m)",
        "space": "O(m + n)",
        "notes": "m=hyperedges, n=nodes; per-order binomial tests with FDR.",
    },
    "statistically_validated_cores": {
        "time": "O(sum_e 2^|e|)",
        "space": "O(G)",
        "notes": "Enumerates sub-groups per order; G=number of distinct sub-groups.",
    },
}

append_complexity_docstrings(globals(), _COMPLEXITY)
