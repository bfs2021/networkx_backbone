"""
Structural backbone extraction methods.

These methods use the network's topology (edge weights, shortest paths,
spanning trees, degree, neighborhood overlap) to identify the most
important substructure.

Methods
-------
global_threshold_filter
    Simple weight cutoff.
strongest_n_ties
    Per-node strongest edges.
global_sparsification
    Keep top-weight edges globally by fraction.
primary_linkage_analysis
    Nystuen & Dacey (1961) -- strongest outgoing edge per node.
edge_betweenness_filter
    Girvan & Newman (2002) -- keep top edge-betweenness edges.
node_degree_filter
    Keep nodes above a minimum degree threshold.
high_salience_skeleton
    Grady et al. (2012) -- shortest-path tree participation.
metric_backbone
    Simas et al. (2021) -- edges on shortest paths (sum distances).
ultrametric_backbone
    Shortest paths using max distance.
doubly_stochastic_filter
    Slater (2009) -- Sinkhorn-Knopp normalization.
h_backbone
    Zhang et al. (2018) -- h-index inspired.
modularity_backbone
    Rajeh et al. (2022) -- node vitality index.
planar_maximally_filtered_graph
    Tumminello et al. (2005) -- planar constraint.
maximum_spanning_tree_backbone
    Maximum spanning tree.
"""

import heapq
import math
import warnings

import networkx as nx
from networkx.utils import not_implemented_for

from networkx_backbone._docstrings import append_complexity_docstrings

__all__ = [
    "global_threshold_filter",
    "strongest_n_ties",
    "global_sparsification",
    "primary_linkage_analysis",
    "edge_betweenness_filter",
    "node_degree_filter",
    "high_salience_skeleton",
    "metric_backbone",
    "ultrametric_backbone",
    "doubly_stochastic_filter",
    "h_backbone",
    "modularity_backbone",
    "planar_maximally_filtered_graph",
    "maximum_spanning_tree_backbone",
]


def _edge_key(G, u, v):
    """Return a deterministic edge key that is direction-aware."""
    if G.is_directed():
        return (u, v)
    ku = (type(u).__name__, repr(u))
    kv = (type(v).__name__, repr(v))
    return (u, v) if ku <= kv else (v, u)


def _check_weights(G, weight, func_name, sign=None):
    """Validate that every edge carries a numeric *weight*.

    Parameters
    ----------
    sign : {None, "nonnegative", "positive"}
        Additional constraint on the weight values.

    Raises
    ------
    NetworkXError
        If an edge has no *weight* attribute, a non-numeric or NaN weight,
        or a weight violating *sign*.
    """
    for u, v, data in G.edges(data=True):
        if weight not in data:
            raise nx.NetworkXError(
                f"{func_name}: edge ({u!r}, {v!r}) has no {weight!r} attribute; "
                f"every edge needs a numeric weight (set one, or pass the name "
                f"of the weight attribute via `weight=`)"
            )
        w = data[weight]
        try:
            w = float(w)
        except (TypeError, ValueError):
            raise nx.NetworkXError(
                f"{func_name}: edge ({u!r}, {v!r}) has non-numeric weight {w!r}"
            ) from None
        if math.isnan(w):
            raise nx.NetworkXError(f"{func_name}: edge ({u!r}, {v!r}) has NaN weight")
        if sign == "positive" and w <= 0:
            raise nx.NetworkXError(
                f"{func_name}: edge ({u!r}, {v!r}) has weight {w!r}; weights must "
                f"be strictly positive (they are converted to distances 1/weight)"
            )
        if sign == "nonnegative" and w < 0:
            raise nx.NetworkXError(
                f"{func_name}: edge ({u!r}, {v!r}) has negative weight {w!r}"
            )


def _caller_stacklevel():
    """``warnings.warn`` stacklevel pointing at the user's call site.

    Skips the NetworkX decorator (argmap) frames that wrap the public
    function; their number differs between the first and later calls.
    """
    import sys

    decorators_file = nx.utils.decorators.__file__
    # Frame 0 is this helper and frame 1 the public function, which is
    # stacklevel 1 for its ``warnings.warn`` call; frame 2 is stacklevel 2.
    frame = sys._getframe(2)
    level = 2
    while frame is not None and (
        frame.f_code.co_filename == decorators_file
        or "argmap" in frame.f_code.co_filename
    ):
        frame = frame.f_back
        level += 1
    return level


def _h_index(values):
    """Largest integer h such that at least h of *values* are >= h."""
    h = 0
    for i, val in enumerate(sorted(values, reverse=True), start=1):
        if _at_least(val, i):
            h = i
        else:
            break
    return h


def _at_least(x, h):
    """``x >= h`` with a relative tolerance for floating-point sums."""
    return x >= h or math.isclose(x, h, rel_tol=1e-9)


def _mark_kept_edges(H, keep_edges, attr):
    """Annotate every edge in H with a boolean keep flag."""
    for u, v, data in H.edges(data=True):
        data[attr] = _edge_key(H, u, v) in keep_edges


# =====================================================================
# 1. Global threshold filter
# =====================================================================


@not_implemented_for("multigraph")
def global_threshold_filter(G, threshold, weight="weight"):
    """Score edges against a global weight threshold.

    For each edge, mark whether its *weight* attribute is greater than
    or equal to *threshold*.

    Parameters
    ----------
    G : graph
        A NetworkX graph.
    threshold : float
        Minimum edge weight required.  Edges with
        ``data[weight] >= threshold`` are kept.
    weight : string, optional (default="weight")
        Edge attribute key used as the weight.

    Returns
    -------
    H : graph
        A copy of *G* with a boolean ``"global_threshold_keep"`` edge
        attribute.

    Raises
    ------
    NetworkXNotImplemented
        If *G* is a multigraph.  Collapse parallel edges first with
        :func:`~networkx_backbone.multigraph_to_weighted`.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import boolean_filter, global_threshold_filter
    >>> G = nx.les_miserables_graph()
    >>> H = global_threshold_filter(G, threshold=3.0)
    >>> backbone = boolean_filter(H, "global_threshold_keep")
    >>> backbone.number_of_edges() <= H.number_of_edges()
    True
    """
    H = G.copy()
    for _, _, data in H.edges(data=True):
        data["global_threshold_keep"] = data.get(weight, 0) >= threshold
    return H


# =====================================================================
# 2. Per-node strongest-N ties
# =====================================================================


@not_implemented_for("multigraph")
def strongest_n_ties(G, n=1, weight="weight"):
    """Score edges by strongest-*n* membership per node.

    For each node, select the *n* incident edges with the largest
    *weight* value.  An edge is kept if *either* endpoint selects it
    (OR semantics).  For directed graphs the selection is based on
    out-edges.

    Parameters
    ----------
    G : graph
        A NetworkX graph.
    n : int, optional (default=1)
        Number of strongest edges to keep per node.  Must be at least 1.
    weight : string, optional (default="weight")
        Edge attribute key used as the weight.

    Returns
    -------
    H : graph
        A copy of *G* with a boolean ``"strongest_n_ties_keep"`` edge
        attribute.

    Raises
    ------
    ValueError
        If *n* < 1.
    NetworkXNotImplemented
        If *G* is a multigraph.  Collapse parallel edges first with
        :func:`~networkx_backbone.multigraph_to_weighted`.

    Notes
    -----
    Ties in weight are broken by neighbour iteration order, so exactly
    *n* edges are selected per node (fewer if the node has fewer edges).
    Edges without a *weight* attribute count as weight 0.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import boolean_filter, strongest_n_ties
    >>> G = nx.les_miserables_graph()
    >>> H = strongest_n_ties(G, n=1)
    >>> backbone = boolean_filter(H, "strongest_n_ties_keep")
    >>> backbone.number_of_edges() <= H.number_of_edges()
    True
    """
    if n < 1:
        raise ValueError("n must be >= 1")

    kept_edges = set()
    # G[u] holds the successors for directed graphs and the neighbours
    # otherwise.  Sort on the weight only so that mixed-type node labels
    # are never compared.
    for u in G.nodes():
        nbrs = [(data.get(weight, 0), v) for v, data in G[u].items()]
        for _, v in heapq.nlargest(n, nbrs, key=lambda x: x[0]):
            kept_edges.add(_edge_key(G, u, v))

    H = G.copy()
    _mark_kept_edges(H, kept_edges, "strongest_n_ties_keep")
    return H


# =====================================================================
# 2b. Global sparsification
# =====================================================================


@not_implemented_for("multigraph")
def global_sparsification(G, s=0.5, weight="weight"):
    """Score edges by global top-fraction membership.

    This method ranks all edges by *weight* and keeps the top ``s``
    fraction (``ceil(s * m)`` edges, at least one).  It is a global
    rank-based weight threshold.

    Parameters
    ----------
    G : graph
        A NetworkX graph.
    s : float, optional (default=0.5)
        Fraction of edges to retain. Must be in (0, 1].
    weight : string, optional (default="weight")
        Edge attribute key used as the weight.

    Returns
    -------
    H : graph
        A copy of *G* with a boolean ``"global_sparsification_keep"``
        edge attribute.

    Raises
    ------
    ValueError
        If ``s`` is outside (0, 1].
    NetworkXNotImplemented
        If *G* is a multigraph.  Collapse parallel edges first with
        :func:`~networkx_backbone.multigraph_to_weighted`.

    Notes
    -----
    Satuluri et al. (2011) [1]_ use the same global top-fraction rule as a
    baseline ("global sparsification"), but rank edges by the Jaccard
    similarity of their endpoints rather than by weight.  To reproduce
    that baseline, score the edges with
    :func:`~networkx_backbone.jaccard_backbone` and pass
    ``weight="jaccard"``.  Ties are broken by the string form of the
    endpoints; edges without a *weight* attribute count as weight 0.

    References
    ----------
    .. [1] Satuluri, V., Parthasarathy, S., & Ruan, Y. (2011). Local graph
       sparsification for scalable clustering. *SIGMOD*, 721-732.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import boolean_filter, global_sparsification
    >>> G = nx.Graph()
    >>> G.add_weighted_edges_from([(0, 1, 5.0), (1, 2, 1.0), (2, 3, 4.0), (3, 0, 2.0)])
    >>> H = global_sparsification(G, s=0.5)
    >>> sorted(boolean_filter(H, "global_sparsification_keep").edges())
    [(0, 1), (2, 3)]
    """
    if not 0.0 < s <= 1.0:
        raise ValueError("s must be in (0, 1]")

    H = G.copy()
    m = G.number_of_edges()
    if m == 0:
        return H

    keep = max(1, math.ceil(s * m))
    ranked_edges = sorted(
        G.edges(data=True),
        key=lambda x: (x[2].get(weight, 0.0), str(x[0]), str(x[1])),
        reverse=True,
    )
    keep_edges = {_edge_key(G, u, v) for u, v, _ in ranked_edges[:keep]}
    _mark_kept_edges(H, keep_edges, "global_sparsification_keep")
    return H


# =====================================================================
# 2c. Primary linkage analysis
# =====================================================================


@not_implemented_for("multigraph")
def primary_linkage_analysis(G, weight="weight"):
    """Score edges by primary-linkage membership.

    Each node keeps only its strongest outgoing connection (its strongest
    incident edge for undirected graphs).  An edge is kept if it is the
    primary link of at least one endpoint.

    Parameters
    ----------
    G : networkx.Graph or networkx.DiGraph
        Input weighted graph.
    weight : string, optional (default="weight")
        Edge attribute key used as the weight.

    Returns
    -------
    H : graph
        A copy of *G* with a boolean ``"primary_linkage_keep"`` edge
        attribute.

    Raises
    ------
    NetworkXNotImplemented
        If *G* is a multigraph.  Collapse parallel edges first with
        :func:`~networkx_backbone.multigraph_to_weighted`.

    Notes
    -----
    Ties are broken by the string form of the neighbour (largest wins);
    edges without a *weight* attribute count as weight 0.

    References
    ----------
    .. [1] Nystuen, J. D., & Dacey, M. F. (1961). A graph theory
       interpretation of nodal regions. *Papers in Regional Science*,
       7(1), 29-42.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import boolean_filter, primary_linkage_analysis
    >>> G = nx.DiGraph()
    >>> G.add_weighted_edges_from([("a", "b", 3), ("a", "c", 1), ("b", "c", 2), ("c", "a", 5)])
    >>> H = primary_linkage_analysis(G)
    >>> sorted(boolean_filter(H, "primary_linkage_keep").edges())
    [('a', 'b'), ('b', 'c'), ('c', 'a')]
    """
    H = G.copy()
    kept_edges = set()

    if G.is_directed():
        for u in G.nodes():
            out_edges = list(G.out_edges(u, data=True))
            if not out_edges:
                continue
            _, v, _ = max(out_edges, key=lambda x: (x[2].get(weight, 0.0), str(x[1])))
            kept_edges.add(_edge_key(G, u, v))
        _mark_kept_edges(H, kept_edges, "primary_linkage_keep")
        return H

    for u in G.nodes():
        neighbors = list(G[u].items())
        if not neighbors:
            continue
        v, _ = max(neighbors, key=lambda x: (x[1].get(weight, 0.0), str(x[0])))
        kept_edges.add(_edge_key(G, u, v))

    _mark_kept_edges(H, kept_edges, "primary_linkage_keep")
    return H


# =====================================================================
# 2d. Edge betweenness filter
# =====================================================================


@not_implemented_for("multigraph")
def edge_betweenness_filter(G, s=0.5, weight="weight"):
    """Score edges by edge betweenness and top-fraction membership.

    Parameters
    ----------
    G : graph
        A NetworkX graph.
    s : float, optional (default=0.5)
        Fraction of edges to retain. Must be in (0, 1].
    weight : string, optional (default="weight")
        Edge attribute key used as the weight. If present, distance is
        interpreted as ``1 / weight``.

    Returns
    -------
    H : graph
        A copy of *G* with ``"edge_betweenness"`` scores and a boolean
        ``"edge_betweenness_keep"`` edge attribute.

    Raises
    ------
    ValueError
        If ``s`` is outside (0, 1].
    NetworkXNotImplemented
        If *G* is a multigraph.  Collapse parallel edges first with
        :func:`~networkx_backbone.multigraph_to_weighted`.

    Notes
    -----
    If no edge has a *weight* attribute, hop-count shortest paths are
    used.  Edges with non-positive weight get infinite distance.

    References
    ----------
    .. [1] Girvan, M., & Newman, M. E. J. (2002). Community structure in
       social and biological networks. *PNAS*, 99(12), 7821-7826.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import boolean_filter, edge_betweenness_filter
    >>> G = nx.barbell_graph(3, 0)  # two triangles joined by the bridge (2, 3)
    >>> H = edge_betweenness_filter(G, s=0.1)
    >>> sorted(boolean_filter(H, "edge_betweenness_keep").edges())
    [(2, 3)]
    """
    if not 0.0 < s <= 1.0:
        raise ValueError("s must be in (0, 1]")

    scored = G.copy()
    dist_attr = "__eb_dist__"
    use_weight = any(weight in data for _, _, data in scored.edges(data=True))

    if use_weight:
        for _, _, data in scored.edges(data=True):
            w = data.get(weight, 1.0)
            data[dist_attr] = 1.0 / w if w > 0 else float("inf")
        eb = nx.edge_betweenness_centrality(scored, weight=dist_attr, normalized=True)
    else:
        eb = nx.edge_betweenness_centrality(scored, weight=None, normalized=True)

    for u, v, data in scored.edges(data=True):
        data["edge_betweenness"] = float(eb.get((u, v), eb.get((v, u), 0.0)))
        data.pop(dist_attr, None)

    m = scored.number_of_edges()
    keep = max(1, math.ceil(s * m)) if m > 0 else 0
    ranked_edges = sorted(
        scored.edges(data=True),
        key=lambda x: (x[2]["edge_betweenness"], str(x[0]), str(x[1])),
        reverse=True,
    )
    keep_edges = {_edge_key(scored, u, v) for u, v, _ in ranked_edges[:keep]}
    _mark_kept_edges(scored, keep_edges, "edge_betweenness_keep")
    return scored


# =====================================================================
# 2e. Node-degree filter
# =====================================================================


@not_implemented_for("multigraph")
def node_degree_filter(G, min_degree=1):
    """Score nodes and edges by a minimum node-degree criterion.

    Parameters
    ----------
    G : graph
        A NetworkX graph.
    min_degree : int, optional (default=1)
        Minimum degree required for a node to be retained.

    Returns
    -------
    H : graph
        A copy of *G* with boolean ``"node_degree_keep"`` on nodes and
        edges.

    Raises
    ------
    ValueError
        If ``min_degree`` is negative.
    NetworkXNotImplemented
        If *G* is a multigraph.  Collapse parallel edges first with
        :func:`~networkx_backbone.multigraph_to_weighted`.

    References
    ----------
    .. [1] Freeman, L. C. (1978). Centrality in social networks:
       conceptual clarification. *Social Networks*, 1(3), 215-239.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import boolean_filter, node_degree_filter
    >>> G = nx.path_graph(4)
    >>> H = node_degree_filter(G, min_degree=2)
    >>> [n for n, keep in H.nodes(data="node_degree_keep") if keep]
    [1, 2]
    >>> sorted(boolean_filter(H, "node_degree_keep").edges())
    [(1, 2)]
    """
    if min_degree < 0:
        raise ValueError("min_degree must be >= 0")
    H = G.copy()
    keep_nodes = {n for n, deg in G.degree() if deg >= min_degree}
    for n in H.nodes():
        H.nodes[n]["node_degree_keep"] = n in keep_nodes
    for u, v, data in H.edges(data=True):
        data["node_degree_keep"] = (u in keep_nodes) and (v in keep_nodes)
    return H


# =====================================================================
# 3. High salience skeleton -- Grady et al. (2012)
# =====================================================================


@not_implemented_for("directed")
@not_implemented_for("multigraph")
def high_salience_skeleton(G, weight="weight"):
    """Compute edge salience from shortest-path tree participation.

    For every node *r*, a shortest-path tree rooted at *r* is computed
    (using inverse weights as distances).  The salience of an edge is the
    fraction of these trees that contain it [1]_.  The salience is stored
    as the ``"salience"`` edge attribute on the returned graph.

    Parameters
    ----------
    G : graph
        An undirected NetworkX graph.  All weights must be positive.
    weight : string, optional (default="weight")
        Edge attribute key used as the weight.

    Returns
    -------
    H : graph
        A copy of *G* with an additional ``"salience"`` edge attribute
        whose value lies in [0, 1].

    Raises
    ------
    NetworkXError
        If an edge has no *weight* attribute or a non-positive weight.
    NetworkXNotImplemented
        If *G* is directed or a multigraph.  Collapse parallel edges first
        with :func:`~networkx_backbone.multigraph_to_weighted`.

    Notes
    -----
    Grady et al. use one shortest-path tree per root.  When a node *x* is
    reached from the root by several equally short paths through *k*
    different predecessors, the tree is not unique; here each of those
    *k* tree edges receives credit ``1/k`` for that root.  This equals
    the expected salience when every node picks one of its shortest-path
    predecessors uniformly at random, and it does not depend on node
    order.  Consequently every root contributes exactly one unit of
    credit per node it reaches, and for a connected graph on *n* nodes the
    saliences sum to ``n - 1``.  Exact ties are detected with
    floating-point equality of path lengths.

    References
    ----------
    .. [1] Grady, D., Thiemann, C., & Brockmann, D. (2012). Robust
       classification of salient links in complex networks. *Nature
       Communications*, 3, 864.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import high_salience_skeleton, threshold_filter
    >>> G = nx.les_miserables_graph()
    >>> H = high_salience_skeleton(G)
    >>> backbone = threshold_filter(H, "salience", 0.5, mode="above")
    >>> backbone.number_of_edges() <= H.number_of_edges()
    True
    """
    _check_weights(G, weight, "high_salience_skeleton", sign="positive")
    H = G.copy()
    n = G.number_of_nodes()
    credit = {_edge_key(G, u, v): 0.0 for u, v in G.edges()}

    def distance(u, v, data):
        return 1.0 / data[weight]

    for root in G.nodes():
        pred, _ = nx.dijkstra_predecessor_and_distance(G, root, weight=distance)
        for node, preds in pred.items():
            if not preds:
                continue
            share = 1.0 / len(preds)
            for p in preds:
                credit[_edge_key(G, node, p)] += share

    for u, v, data in H.edges(data=True):
        data["salience"] = credit[_edge_key(G, u, v)] / n

    return H


# =====================================================================
# 4/5. Metric & ultrametric backbones -- Simas et al. (2021)
# =====================================================================


@not_implemented_for("directed")
@not_implemented_for("multigraph")
def metric_backbone(G, weight="weight"):
    """Extract the metric backbone using sum-of-distances shortest paths.

    An edge *(u, v)* is in the metric backbone if and only if its direct
    distance (the inverse of its weight) equals the shortest-path
    distance between *u* and *v* computed over the full graph.

    Parameters
    ----------
    G : graph
        An undirected NetworkX graph.
    weight : string, optional (default="weight")
        Edge attribute key used as the weight.  All weights must be
        positive.

    Returns
    -------
    H : graph
        A copy of *G* with boolean ``"metric_keep"`` on each edge.

    Raises
    ------
    NetworkXError
        If an edge has no *weight* attribute or a non-positive weight.
    NetworkXNotImplemented
        If *G* is directed or a multigraph.  Collapse parallel edges first
        with :func:`~networkx_backbone.multigraph_to_weighted`.

    Notes
    -----
    Distances are compared with a relative tolerance of ``1e-9`` so that
    an edge tied with an alternative path is kept regardless of rounding.
    Self-loops are never on a shortest path and are never kept.

    References
    ----------
    .. [1] Simas, T., Correia, R. B., & Rocha, L. M. (2021). The distance
       backbone of complex networks. *J. Complex Netw.*, 9, cnab021.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import boolean_filter, metric_backbone
    >>> G = nx.les_miserables_graph()
    >>> H = metric_backbone(G)
    >>> backbone = boolean_filter(H, "metric_keep")
    >>> backbone.number_of_edges() <= H.number_of_edges()
    True
    """
    return _distance_backbone(G, weight=weight, ultrametric=False)


@not_implemented_for("directed")
@not_implemented_for("multigraph")
def ultrametric_backbone(G, weight="weight"):
    """Extract the ultrametric backbone using max-distance (minimax) paths.

    An edge *(u, v)* is in the ultrametric backbone if its direct
    distance is no greater than the minimax path distance between *u*
    and *v* (the path that minimises the maximum edge distance along
    the path).

    Parameters
    ----------
    G : graph
        An undirected NetworkX graph.
    weight : string, optional (default="weight")
        Edge attribute key used as the weight.  All weights must be
        positive.

    Returns
    -------
    H : graph
        A copy of *G* with boolean ``"ultrametric_keep"`` on each edge.

    Raises
    ------
    NetworkXError
        If an edge has no *weight* attribute or a non-positive weight.
    NetworkXNotImplemented
        If *G* is directed or a multigraph.  Collapse parallel edges first
        with :func:`~networkx_backbone.multigraph_to_weighted`.

    Notes
    -----
    Minimax distances are read off a minimum spanning forest of the
    distance graph and compared with a relative tolerance of ``1e-9``.
    Self-loops are never kept.

    References
    ----------
    .. [1] Simas, T., Correia, R. B., & Rocha, L. M. (2021). The distance
       backbone of complex networks. *J. Complex Netw.*, 9, cnab021.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import boolean_filter, ultrametric_backbone
    >>> G = nx.les_miserables_graph()
    >>> H = ultrametric_backbone(G)
    >>> backbone = boolean_filter(H, "ultrametric_keep")
    >>> backbone.number_of_edges() <= H.number_of_edges()
    True
    """
    return _distance_backbone(G, weight=weight, ultrametric=True)


def _distance_backbone(G, weight="weight", ultrametric=False):
    """Shared helper for metric and ultrametric backbones."""
    func_name = "ultrametric_backbone" if ultrametric else "metric_backbone"
    _check_weights(G, weight, func_name, sign="positive")
    H = G.copy()

    dist_G = nx.Graph()
    dist_G.add_nodes_from(G)
    for u, v, data in G.edges(data=True):
        if u != v:
            dist_G.add_edge(u, v, dist=1.0 / data[weight])

    keep_edges = set()
    if ultrametric:
        mst = nx.minimum_spanning_tree(dist_G, weight="dist")
        for u, v, data in G.edges(data=True):
            if u == v:
                continue
            direct_dist = 1.0 / data[weight]
            path = nx.shortest_path(mst, u, v, weight=None)
            max_edge = max(
                dist_G[path[i]][path[i + 1]]["dist"] for i in range(len(path) - 1)
            )
            if direct_dist <= max_edge or math.isclose(
                direct_dist, max_edge, rel_tol=1e-9
            ):
                keep_edges.add(_edge_key(G, u, v))
    else:
        sp = dict(nx.all_pairs_dijkstra_path_length(dist_G, weight="dist"))
        for u, v, data in G.edges(data=True):
            if u == v:
                continue
            direct_dist = 1.0 / data[weight]
            if math.isclose(direct_dist, sp[u][v], rel_tol=1e-9):
                keep_edges.add(_edge_key(G, u, v))

    keep_attr = "ultrametric_keep" if ultrametric else "metric_keep"
    _mark_kept_edges(H, keep_edges, keep_attr)
    return H


# =====================================================================
# 6. Doubly stochastic filter -- Slater (2009)
# =====================================================================


@not_implemented_for("directed")
@not_implemented_for("multigraph")
def doubly_stochastic_filter(G, weight="weight", max_iter=1000, tol=1e-8):
    """Compute the doubly-stochastic backbone via Sinkhorn-Knopp normalization.

    Transforms the adjacency matrix into a doubly stochastic matrix via
    iterative Sinkhorn-Knopp normalization, then stores the normalised
    weight as the ``"ds_weight"`` edge attribute on the returned graph.

    Parameters
    ----------
    G : graph
        An undirected NetworkX graph.
    weight : string, optional (default="weight")
        Edge attribute key used as the weight.  Weights must be
        non-negative.
    max_iter : int, optional (default=1000)
        Maximum number of Sinkhorn-Knopp iterations.
    tol : float, optional (default=1e-8)
        Convergence tolerance on the row and column sums.

    Returns
    -------
    H : graph
        A copy of *G* with an additional ``"ds_weight"`` edge attribute
        (doubly-stochastic normalised weight).

    Raises
    ------
    NetworkXError
        If an edge has no *weight* attribute or a negative weight.
    NetworkXNotImplemented
        If *G* is directed or a multigraph.  Collapse parallel edges first
        with :func:`~networkx_backbone.multigraph_to_weighted`.

    Warns
    -----
    RuntimeWarning
        If the iteration does not converge within *max_iter* steps.  This
        happens when the weighted adjacency matrix cannot be scaled to a
        doubly stochastic matrix (it lacks *total support*), for example
        for trees such as a path on three nodes or any graph in which two
        leaves hang off the same node, or when convergence is too slow
        (common for sparse real networks with low-degree nodes).  ``"ds_weight"`` then
        holds the last iterate, which is not doubly stochastic.

    Notes
    -----
    Isolated nodes (all-zero rows) are ignored when checking convergence.

    References
    ----------
    .. [1] Slater, P. B. (2009). A two-stage algorithm for extracting
       the multiscale backbone of complex weighted networks. *PNAS*,
       106(26), E66.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import doubly_stochastic_filter, threshold_filter
    >>> G = nx.complete_graph(4)
    >>> weights = dict(zip(G.edges(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    >>> nx.set_edge_attributes(G, weights, "weight")
    >>> H = doubly_stochastic_filter(G)
    >>> row_sum = sum(d["ds_weight"] for _, _, d in H.edges(0, data=True))
    >>> round(row_sum, 6)
    1.0
    >>> backbone = threshold_filter(H, "ds_weight", 0.36, mode="above")
    >>> sorted(backbone.edges())
    [(0, 3), (1, 2)]
    """
    import numpy as np

    _check_weights(G, weight, "doubly_stochastic_filter", sign="nonnegative")
    nodes = list(G)
    n = len(nodes)
    if n == 0:
        return G.copy()

    idx = {v: i for i, v in enumerate(nodes)}

    # Build adjacency matrix
    A = np.zeros((n, n), dtype=float)
    for u, v, data in G.edges(data=True):
        w = float(data[weight])
        A[idx[u], idx[v]] = w
        A[idx[v], idx[u]] = w

    active = A.sum(axis=1) > 0  # symmetric: same for rows and columns

    # Sinkhorn-Knopp iteration
    D = A.copy()
    converged = False
    for _ in range(max_iter):
        # Row normalise
        row_sums = D.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        D = D / row_sums
        # Column normalise
        col_sums = D.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1.0
        D = D / col_sums
        # Check convergence on the non-empty rows/columns
        rs = D.sum(axis=1)[active]
        cs = D.sum(axis=0)[active]
        if np.allclose(rs, 1, rtol=0, atol=tol) and np.allclose(
            cs, 1, rtol=0, atol=tol
        ):
            converged = True
            break

    if not converged:
        warnings.warn(
            "doubly_stochastic_filter: Sinkhorn-Knopp normalization did not "
            f"converge within max_iter={max_iter} iterations; the weighted "
            "adjacency matrix probably cannot be scaled to a doubly stochastic "
            "matrix (e.g. trees, or several leaves attached to one node). "
            "'ds_weight' holds the last, non-doubly-stochastic iterate.",
            RuntimeWarning,
            stacklevel=_caller_stacklevel(),
        )

    H = G.copy()
    for u, v, data in H.edges(data=True):
        data["ds_weight"] = float(D[idx[u], idx[v]])

    return H


# =====================================================================
# 7. h-backbone -- Zhang et al. (2018)
# =====================================================================


@not_implemented_for("directed")
@not_implemented_for("multigraph")
def h_backbone(G, weight="weight"):
    r"""Extract the h-backbone of a weighted graph.

    Following Zhang et al. (2018) [1]_, the h-backbone is the union of

    1. the **h-strength network**: the edges whose weight is at least the
       *h-strength* :math:`h_s`, the largest integer such that at least
       :math:`h_s` edges have weight :math:`\ge h_s`; and
    2. the **h-bridge network**: the edges whose *bridge*
       :math:`b(e) = eb(e) / N` is at least the *h-bridge* :math:`h_b`, the
       largest integer such that at least :math:`h_b` edges have
       :math:`b(e) \ge h_b`.  Here :math:`eb(e)` is the (unnormalised)
       edge betweenness of *e* counted over ordered node pairs and *N* is
       the number of nodes.

    Parameters
    ----------
    G : graph
        An undirected NetworkX graph.
    weight : string, optional (default="weight")
        Edge attribute key used as the weight.

    Returns
    -------
    H : graph
        A copy of *G* with the float ``"h_bridge"`` (the bridge value
        :math:`b(e)`) and boolean ``"h_backbone_keep"`` on each edge.

    Raises
    ------
    NetworkXError
        If an edge has no *weight* attribute or a non-numeric weight.
    NetworkXNotImplemented
        If *G* is directed or a multigraph.  Collapse parallel edges first
        with :func:`~networkx_backbone.multigraph_to_weighted`.

    Notes
    -----
    As in [1]_, edge betweenness is computed on the whole (unweighted)
    network topology: the h-bridge captures structural importance while
    the weights enter only through the h-strength.  If every weight is
    below 1 then :math:`h_s = 0` and the h-strength network is empty (it
    would otherwise contain every edge); likewise an h-bridge of 0
    contributes no edges.  Bridge values are compared with a relative
    tolerance of ``1e-9``.

    References
    ----------
    .. [1] Zhang, R. J., Stanley, H. E., & Ye, F. Y. (2018). Extracting
       h-backbone as a core structure in weighted networks. *Scientific
       Reports*, 8, 14356.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import boolean_filter, h_backbone
    >>> G = nx.les_miserables_graph()
    >>> H = h_backbone(G)
    >>> backbone = boolean_filter(H, "h_backbone_keep")
    >>> backbone.number_of_edges() <= H.number_of_edges()
    True
    """
    _check_weights(G, weight, "h_backbone")
    H = G.copy()
    n = G.number_of_nodes()

    # h-strength
    h_s = _h_index(float(d[weight]) for _, _, d in G.edges(data=True))

    # h-bridge: b(e) = eb(e) / N with eb over ordered pairs (NetworkX counts
    # unordered pairs for undirected graphs, hence the factor 2).
    eb = nx.edge_betweenness_centrality(G, normalized=False, weight=None)
    bridge = {}
    for u, v in G.edges():
        val = eb.get((u, v), eb.get((v, u), 0.0))
        bridge[_edge_key(G, u, v)] = 2.0 * val / n
    h_b = _h_index(bridge.values())

    for u, v, data in H.edges(data=True):
        b = bridge[_edge_key(G, u, v)]
        data["h_bridge"] = b
        in_strength = h_s >= 1 and float(data[weight]) >= h_s
        in_bridge = h_b >= 1 and _at_least(b, h_b)
        data["h_backbone_keep"] = in_strength or in_bridge
    return H


# =====================================================================
# 8. Modularity backbone -- Rajeh et al. (2022)
# =====================================================================


@not_implemented_for("directed")
@not_implemented_for("multigraph")
def modularity_backbone(G, weight="weight", seed=None):
    """Compute the modularity vitality index for each node.

    The vitality index [1]_ measures the change in modularity when a
    node is removed.  Nodes that contribute positively to community
    structure have positive vitality.  The vitality is stored as the
    ``"vitality"`` node attribute on the returned graph.

    Parameters
    ----------
    G : graph
        A NetworkX graph.
    weight : string, optional (default="weight")
        Edge attribute key used as the weight.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state, passed unchanged to
        every :func:`~networkx.algorithms.community.louvain.louvain_communities` call.  With an
        integer seed each call is seeded identically, so results are
        reproducible.  See :ref:`Randomness<networkx:randomness>`.

    Returns
    -------
    H : graph
        A copy of *G* with ``"vitality"`` node scores and boolean
        ``"modularity_keep"`` on edges.

    Raises
    ------
    NetworkXNotImplemented
        If *G* is directed or a multigraph.  Collapse parallel edges first
        with :func:`~networkx_backbone.multigraph_to_weighted`.

    Notes
    -----
    Louvain is randomised, so with ``seed=None`` the vitalities (and the
    retained edges) can change between calls.  A graph (or node-deleted
    subgraph) with zero total edge weight has modularity 0.

    References
    ----------
    .. [1] Rajeh, S., Savonnet, M., Leclercq, E., & Cherifi, H. (2022).
       Modularity-based backbone extraction in weighted complex networks.
       *NetSci-X 2022*, 67-79.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import boolean_filter, modularity_backbone
    >>> G = nx.les_miserables_graph()
    >>> H = modularity_backbone(G, seed=42)
    >>> backbone = boolean_filter(H, "modularity_keep")
    >>> backbone.number_of_edges() <= H.number_of_edges()
    True
    """
    H = G.copy()

    def louvain_modularity(graph):
        if graph.size(weight=weight) == 0:
            return 0.0
        communities = nx.community.louvain_communities(graph, weight=weight, seed=seed)
        return nx.community.modularity(graph, communities, weight=weight)

    mod_full = louvain_modularity(G)

    for node in G.nodes():
        G_reduced = G.copy()
        G_reduced.remove_node(node)
        H.nodes[node]["vitality"] = mod_full - louvain_modularity(G_reduced)

    for u, v, data in H.edges(data=True):
        data["modularity_keep"] = (
            H.nodes[u].get("vitality", 0.0) > 0.0
            and H.nodes[v].get("vitality", 0.0) > 0.0
        )

    return H


# =====================================================================
# 9. Planar Maximally Filtered Graph -- Tumminello et al. (2005)
# =====================================================================


@not_implemented_for("directed")
@not_implemented_for("multigraph")
def planar_maximally_filtered_graph(G, weight="weight"):
    """Construct the Planar Maximally Filtered Graph (PMFG).

    Iteratively adds edges from heaviest to lightest, keeping only
    those whose addition preserves planarity [1]_.

    Parameters
    ----------
    G : graph
        An undirected NetworkX graph.
    weight : string, optional (default="weight")
        Edge attribute key used as the weight (may be negative, e.g.
        correlations).

    Returns
    -------
    H : graph
        A copy of *G* with boolean ``"pmfg_keep"`` on edges.

    Raises
    ------
    NetworkXError
        If an edge has no *weight* attribute or a non-numeric weight.
    NetworkXNotImplemented
        If *G* is directed or a multigraph.  Collapse parallel edges first
        with :func:`~networkx_backbone.multigraph_to_weighted`.

    References
    ----------
    .. [1] Tumminello, M., Aste, T., Di Matteo, T., & Mantegna, R. N.
       (2005). A tool for filtering information in complex systems.
       *PNAS*, 102(30), 10421-10426.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import boolean_filter, planar_maximally_filtered_graph
    >>> G = nx.les_miserables_graph()
    >>> H = planar_maximally_filtered_graph(G)
    >>> backbone = boolean_filter(H, "pmfg_keep")
    >>> backbone.number_of_edges() <= H.number_of_edges()
    True
    """
    _check_weights(G, weight, "planar_maximally_filtered_graph")
    n = G.number_of_nodes()
    max_edges = 3 * (n - 2) if n >= 3 else n - 1

    # Sort edges by weight descending
    edges = sorted(G.edges(data=True), key=lambda e: float(e[2][weight]), reverse=True)

    candidate = G.__class__()
    candidate.add_nodes_from(G.nodes(data=True))

    for u, v, data in edges:
        if candidate.number_of_edges() >= max_edges:
            break
        candidate.add_edge(u, v, **data)
        is_planar, _ = nx.check_planarity(candidate)
        if not is_planar:
            candidate.remove_edge(u, v)

    keep_edges = {_edge_key(candidate, u, v) for u, v in candidate.edges()}
    H = G.copy()
    _mark_kept_edges(H, keep_edges, "pmfg_keep")
    return H


# =====================================================================
# 10. Maximum spanning tree backbone
# =====================================================================


@not_implemented_for("directed")
@not_implemented_for("multigraph")
def maximum_spanning_tree_backbone(G, weight="weight"):
    """Extract the maximum spanning tree as a backbone.

    Parameters
    ----------
    G : graph
        A NetworkX graph.
    weight : string, optional (default="weight")
        Edge attribute key used as the weight.

    Returns
    -------
    H : graph
        A copy of *G* with boolean ``"mst_keep"`` on edges.

    Raises
    ------
    NetworkXNotImplemented
        If *G* is directed or a multigraph.  Collapse parallel edges first
        with :func:`~networkx_backbone.multigraph_to_weighted`.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import boolean_filter, maximum_spanning_tree_backbone
    >>> G = nx.les_miserables_graph()
    >>> H = maximum_spanning_tree_backbone(G)
    >>> backbone = boolean_filter(H, "mst_keep")
    >>> backbone.number_of_edges() == G.number_of_nodes() - 1
    True
    """
    tree = nx.maximum_spanning_tree(G, weight=weight)
    keep_edges = {_edge_key(tree, u, v) for u, v in tree.edges()}
    H = G.copy()
    _mark_kept_edges(H, keep_edges, "mst_keep")
    return H


_COMPLEXITY = {
    "global_threshold_filter": {
        "time": "O(m)",
        "space": "O(n + m)",
        "notes": "n=|V|, m=|E|.",
    },
    "strongest_n_ties": {
        "time": "O(m log n)",
        "space": "O(n + m)",
        "notes": "Worst-case across per-node strongest-edge selection.",
    },
    "global_sparsification": {
        "time": "O(m log m)",
        "space": "O(n + m)",
    },
    "primary_linkage_analysis": {
        "time": "O(m)",
        "space": "O(n + m)",
    },
    "edge_betweenness_filter": {
        "time": "O(nm + n^2 log n)",
        "space": "O(n + m)",
        "notes": "Dominated by weighted Brandes edge-betweenness.",
    },
    "node_degree_filter": {
        "time": "O(n + m)",
        "space": "O(n + m)",
    },
    "high_salience_skeleton": {
        "time": "O(nm + n^2 log n)",
        "space": "O(n + m)",
        "notes": "Runs a shortest-path tree from each root.",
    },
    "metric_backbone": {
        "time": "O(nm + n^2 log n)",
        "space": "O(n^2 + m)",
        "notes": "All-pairs weighted shortest paths.",
    },
    "ultrametric_backbone": {
        "time": "O(mn + m log n)",
        "space": "O(n + m)",
        "notes": "MST plus per-edge minimax path checks.",
    },
    "doubly_stochastic_filter": {
        "time": "O(I * n^2 + m)",
        "space": "O(n^2 + m)",
        "notes": "I=max_iter for Sinkhorn-Knopp normalization.",
    },
    "h_backbone": {
        "time": "O(nm + m log m)",
        "space": "O(n + m)",
        "notes": "Dominated by unweighted edge betweenness (Brandes).",
    },
    "modularity_backbone": {
        "time": "O(nm)",
        "space": "O(n + m)",
        "notes": "Practical/heuristic bound due to repeated Louvain runs.",
    },
    "planar_maximally_filtered_graph": {
        "time": "O(mn + m log m)",
        "space": "O(n + m)",
    },
    "maximum_spanning_tree_backbone": {
        "time": "O(m log n)",
        "space": "O(n + m)",
    },
}

append_complexity_docstrings(globals(), _COMPLEXITY)
