"""
Filtering utilities for backbone extraction.

After a backbone method annotates edges (or nodes) with scores or p-values,
these functions extract the final subgraph.
"""

import math

import networkx as nx

from networkx_backbone._docstrings import append_complexity_docstrings

__all__ = [
    "multigraph_to_weighted",
    "threshold_filter",
    "fraction_filter",
    "boolean_filter",
    "consensus_backbone",
    "adjust_pvalues",
]


_MTC_METHODS = ("none", "bonferroni", "holm", "hochberg", "bh", "fdr", "by")


def adjust_pvalues(pvalues, method="bh"):
    """Apply a multiple-testing correction to a collection of p-values.

    Reproduces R's ``p.adjust`` for the methods used by Neal's *Backbone 3.0*
    ``mtc`` option, so statistical backbone p-values can be corrected for the
    number of edges tested before thresholding.

    Parameters
    ----------
    pvalues : iterable of float
        The raw p-values.
    method : string, optional (default="bh")
        One of ``"none"``, ``"bonferroni"``, ``"holm"`` (step-down),
        ``"hochberg"`` (step-up), ``"bh"``/``"fdr"`` (Benjamini-Hochberg), or
        ``"by"`` (Benjamini-Yekutieli).

    Returns
    -------
    adjusted : list of float
        Adjusted p-values in the same order as the input, each in ``[0, 1]``.
        Missing values (NaN) stay NaN and, as in R's ``p.adjust``, do not
        count toward the number of tests.

    Raises
    ------
    ValueError
        If *method* is not recognised.

    References
    ----------
    .. [1] Holm, S. (1979). A simple sequentially rejective multiple test
       procedure. *Scandinavian Journal of Statistics*, 6(2), 65-70.
    .. [2] Hochberg, Y. (1988). A sharper Bonferroni procedure for multiple
       tests of significance. *Biometrika*, 75(4), 800-802.
    .. [3] Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery
       rate. *J. Royal Statistical Society B*, 57(1), 289-300.
    .. [4] Benjamini, Y., & Yekutieli, D. (2001). The control of the false
       discovery rate in multiple testing under dependency. *Annals of
       Statistics*, 29(4), 1165-1188.

    Examples
    --------
    >>> from networkx_backbone import adjust_pvalues
    >>> adjust_pvalues([0.01, 0.02, 0.5], method="bonferroni")
    [0.03, 0.06, 1.0]
    """
    method_l = method.lower()
    if method_l not in _MTC_METHODS:
        raise ValueError(f"method must be one of {_MTC_METHODS}, got {method!r}")

    raw = [float(x) for x in pvalues]
    finite = [i for i, x in enumerate(raw) if not math.isnan(x)]
    adjusted = [math.nan] * len(raw)
    for i, val in zip(finite, _adjust([raw[i] for i in finite], method_l)):
        adjusted[i] = min(1.0, max(0.0, val))
    return adjusted


def _adjust(p, method_l):
    """Adjust a list of non-missing p-values (helper for adjust_pvalues)."""
    n = len(p)
    if n == 0:
        return []
    if method_l == "none":
        return list(p)
    if method_l == "bonferroni":
        return [n * x for x in p]

    if method_l == "holm":
        # Step-down over ascending p-values with running maximum.
        order = sorted(range(n), key=lambda k: p[k])
        out = [0.0] * n
        running = 0.0
        for rank, idx in enumerate(order):
            running = max(running, (n - rank) * p[idx])
            out[idx] = min(1.0, running)
        return out

    # Step-up methods over descending p-values with running minimum.
    order = sorted(range(n), key=lambda k: p[k], reverse=True)
    out = [0.0] * n
    running = float("inf")
    if method_l == "hochberg":
        for j, idx in enumerate(order):
            running = min(running, (j + 1) * p[idx])
            out[idx] = min(1.0, running)
        return out
    if method_l in ("bh", "fdr"):
        for j, idx in enumerate(order):
            running = min(running, (n / (n - j)) * p[idx])
            out[idx] = min(1.0, running)
        return out
    # Benjamini-Yekutieli
    c = sum(1.0 / k for k in range(1, n + 1))
    for j, idx in enumerate(order):
        running = min(running, c * (n / (n - j)) * p[idx])
        out[idx] = min(1.0, running)
    return out


def multigraph_to_weighted(G, weight="weight", edge_type_attr=None, aggregate="count"):
    """Convert a MultiGraph/MultiDiGraph into a weighted simple graph.

    Parallel edges between the same node pair are collapsed into a single
    weighted edge.

    Parameters
    ----------
    G : graph
        A NetworkX graph. If ``G`` is a ``MultiGraph`` or ``MultiDiGraph``,
        parallel edges are aggregated. For simple graphs, a copy is returned
        with missing weights filled as 1.
    weight : string, optional (default="weight")
        Edge attribute name used to store the aggregated weight.
    edge_type_attr : string or None, optional (default=None)
        If provided, weight is the count of distinct values of this edge
        attribute among parallel edges. If missing on all parallel edges,
        falls back to the number of parallel edges.  Only used with
        ``aggregate="count"``.
    aggregate : {"count", "sum", "max", "min", "mean"}, optional (default="count")
        How parallel edges are combined.  ``"count"`` sets the weight to the
        number of parallel edges (or of distinct edge types).  The other
        options combine the existing *weight* attribute of the parallel edges,
        treating a missing weight as 1.

    Returns
    -------
    H : graph
        ``nx.Graph`` for undirected input and ``nx.DiGraph`` for directed
        input. Each collapsed edge includes:

        - ``weight``: aggregated weight
        - ``"edge_count"``: number of parallel edges collapsed
        - ``"edge_type_count"``: number of distinct edge types (only when
          ``edge_type_attr`` is provided)

    Raises
    ------
    ValueError
        If *aggregate* is not recognized.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import multigraph_to_weighted
    >>> B = nx.davis_southern_women_graph()
    >>> MG = nx.MultiGraph()
    >>> MG.add_nodes_from(B.nodes(data=True))
    >>> u, v = next(iter(B.edges()))
    >>> _ = MG.add_edge(u, v, edge_type="attendance_a")
    >>> _ = MG.add_edge(u, v, edge_type="attendance_b")
    >>> H = multigraph_to_weighted(MG, edge_type_attr="edge_type")
    >>> H[u][v]["weight"]
    2

    Sum existing weights instead of counting edges:

    >>> M = nx.MultiGraph([(0, 1, {"weight": 10}), (0, 1, {"weight": 20})])
    >>> multigraph_to_weighted(M, aggregate="sum")[0][1]["weight"]
    30
    """
    combiners = {
        "sum": sum,
        "max": max,
        "min": min,
        "mean": lambda ws: sum(ws) / len(ws),
    }
    if aggregate != "count" and aggregate not in combiners:
        raise ValueError(
            f"aggregate must be 'count', 'sum', 'max', 'min', or 'mean', got {aggregate!r}"
        )

    if not G.is_multigraph():
        H = G.copy()
        for _, _, data in H.edges(data=True):
            data.setdefault(weight, 1)
        return H

    H = nx.DiGraph() if G.is_directed() else nx.Graph()
    H.graph.update(G.graph)
    H.add_nodes_from(G.nodes(data=True))

    grouped = {}
    for u, v, _, data in G.edges(keys=True, data=True):
        key = (u, v)
        grouped.setdefault(key, []).append(data)

    for (u, v), edge_datas in grouped.items():
        edge_count = len(edge_datas)
        out_data = {"edge_count": edge_count}

        if aggregate != "count":
            out_data[weight] = combiners[aggregate](
                [data.get(weight, 1) for data in edge_datas]
            )
        elif edge_type_attr is None:
            out_data[weight] = edge_count
        else:
            present_types = [
                data[edge_type_attr] for data in edge_datas if edge_type_attr in data
            ]
            type_count = len(set(present_types)) if present_types else edge_count
            out_data[weight] = type_count
            out_data["edge_type_count"] = type_count

        H.add_edge(u, v, **out_data)

    return H


def _empty_like(G, nodes=True):
    """New graph of the same type carrying G's graph (and node) attributes."""
    H = G.__class__()
    H.graph.update(G.graph)
    if nodes:
        H.add_nodes_from(G.nodes(data=True))
    return H


def _edge_iter(G):
    """Yield ``(u, v, key, data)``; ``key`` is None for simple graphs."""
    if G.is_multigraph():
        yield from G.edges(keys=True, data=True)
    else:
        for u, v, data in G.edges(data=True):
            yield u, v, None, data


def _add_edge(H, G, u, v, key, data):
    """Copy one edge (with its key on multigraphs) and its endpoints' data."""
    for node in (u, v):
        if node not in H:
            H.add_node(node, **G.nodes[node])
    if key is None:
        H.add_edge(u, v, **data)
    else:
        H.add_edge(u, v, key=key, **data)


def _is_score(val):
    return val is not None and not (isinstance(val, float) and math.isnan(val))


def threshold_filter(
    G,
    score,
    threshold,
    mode="below",
    filter_on="edges",
    include_all_nodes=True,
    mtc="none",
):
    """Retain edges or nodes whose score passes a threshold test.

    Parameters
    ----------
    G : graph
        A NetworkX graph with a computed score attribute on edges or nodes.
    score : string
        Attribute name to filter on.
    threshold : float
        Cutoff value.
    mode : {"below", "above"}, optional (default="below")
        ``"below"`` keeps elements with ``score < threshold`` (typical for
        p-values).  ``"above"`` keeps elements with ``score >= threshold``
        (typical for salience or importance scores).
    filter_on : {"edges", "nodes"}, optional (default="edges")
        Whether to filter edges or nodes.
    include_all_nodes : bool, optional (default=True)
        Controls isolate retention in the output.

        - If ``filter_on="edges"`` and ``True``, all input nodes are kept.
          If ``False``, only nodes incident to retained edges are kept.
        - If ``filter_on="nodes"`` and ``True``, all retained nodes are kept
          even if isolated in the induced subgraph. If ``False``, retained
          nodes with degree 0 are removed.
    mtc : string, optional (default="none")
        Multiple-testing correction applied to *score* values before
        thresholding (see :func:`adjust_pvalues`).  Only valid with
        ``mode="below"`` (p-values).  Default ``"none"`` leaves scores unchanged.

    Returns
    -------
    H : graph
        Filtered subgraph of the same type as *G*.  When filtering edges,
        all original nodes are preserved by default.  When filtering nodes,
        only retained nodes and their mutual edges are preserved.  Graph,
        node, and edge attributes are copied (as are edge keys of
        multigraphs).  Elements without the *score* attribute (or with a NaN
        score) are dropped.

    See Also
    --------
    fraction_filter : Keep a fixed share of elements instead.
    boolean_filter : Keep edges flagged by a ``*_keep`` attribute.
    adjust_pvalues : The multiple-testing corrections behind ``mtc``.

    Raises
    ------
    ValueError
        If *mode* is not ``"below"`` or ``"above"``, if *filter_on* is
        not ``"edges"`` or ``"nodes"``, or if *mtc* is used with
        ``mode="above"``.

    Examples
    --------
    >>> import networkx as nx

    >>> from networkx_backbone import disparity_filter, threshold_filter
    >>> G = nx.les_miserables_graph()
    >>> H = disparity_filter(G)
    >>> filtered = threshold_filter(H, "disparity_pvalue", 0.5)
    >>> filtered.number_of_edges() <= H.number_of_edges()
    True
    """
    if mode not in ("below", "above"):
        raise ValueError(f"mode must be 'below' or 'above', got {mode!r}")
    if mtc != "none" and mode != "below":
        raise ValueError("mtc correction requires mode='below' (p-values)")

    def _passes(val):
        return (mode == "below" and val < threshold) or (
            mode == "above" and val >= threshold
        )

    if filter_on == "edges":
        H = _empty_like(G, nodes=include_all_nodes)
        scored = [e for e in _edge_iter(G) if e[3].get(score) is not None]
        if mtc != "none":
            values = adjust_pvalues([e[3][score] for e in scored], mtc)
        else:
            values = [e[3][score] for e in scored]
        for (u, v, key, data), val in zip(scored, values):
            if _passes(val):
                _add_edge(H, G, u, v, key, data)
        return H

    elif filter_on == "nodes":
        scored = [
            (node, data)
            for node, data in G.nodes(data=True)
            if data.get(score) is not None
        ]
        if mtc != "none":
            values = adjust_pvalues([data[score] for _, data in scored], mtc)
        else:
            values = [data[score] for _, data in scored]
        keep = {node for (node, _), val in zip(scored, values) if _passes(val)}
        H = G.subgraph(keep).copy()
        if not include_all_nodes:
            isolates = list(nx.isolates(H))
            if isolates:
                H.remove_nodes_from(isolates)
        return H

    else:
        raise ValueError(f"filter_on must be 'edges' or 'nodes', got {filter_on!r}")


def fraction_filter(G, score, fraction, ascending=True, filter_on="edges"):
    """Retain the top or bottom fraction of edges or nodes by score.

    Parameters
    ----------
    G : graph
        A NetworkX graph with a computed score attribute.
    score : string
        Attribute name to sort on.
    fraction : float
        Fraction of elements to retain, in (0, 1].
    ascending : bool, optional (default=True)
        If ``True``, keep the elements with the *smallest* scores (e.g.
        lowest p-values).  If ``False``, keep the *largest*.
    filter_on : {"edges", "nodes"}, optional (default="edges")
        Whether to filter edges or nodes.

    Returns
    -------
    H : graph
        Filtered subgraph of the same type as *G*.  Elements without the
        *score* attribute are never kept; the fraction is taken of the scored
        elements.  Ties are broken by input order.

    Raises
    ------
    ValueError
        If *fraction* is not in (0, 1] or *filter_on* is invalid.

    See Also
    --------
    threshold_filter : Keep elements passing a fixed cutoff instead.

    Examples
    --------
    >>> import networkx as nx

    >>> from networkx_backbone import disparity_filter, fraction_filter
    >>> G = nx.les_miserables_graph()
    >>> H = disparity_filter(G)
    >>> filtered = fraction_filter(H, "disparity_pvalue", 0.5)
    >>> filtered.number_of_edges() <= H.number_of_edges()
    True
    """
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")

    sign = 1 if ascending else -1
    if filter_on == "edges":
        scored = [e for e in _edge_iter(G) if _is_score(e[3].get(score))]
        # A stable sort on the signed score keeps input order among ties.
        scored.sort(key=lambda e: sign * e[3][score])
        n_keep = max(1, int(len(scored) * fraction)) if scored else 0
        H = _empty_like(G)
        for u, v, key, data in scored[:n_keep]:
            _add_edge(H, G, u, v, key, data)
        return H

    elif filter_on == "nodes":
        scored = [
            node for node, data in G.nodes(data=True) if _is_score(data.get(score))
        ]
        scored.sort(key=lambda node: sign * G.nodes[node][score])
        n_keep = max(1, int(len(scored) * fraction)) if scored else 0
        return G.subgraph(scored[:n_keep]).copy()

    else:
        raise ValueError(f"filter_on must be 'edges' or 'nodes', got {filter_on!r}")


def boolean_filter(G, score):
    """Retain edges whose boolean score attribute is truthy.

    Parameters
    ----------
    G : graph
        A NetworkX graph.
    score : string
        Edge attribute name (should contain boolean values).

    Returns
    -------
    H : graph
        A new graph of the same type as *G* containing only edges for
        which ``data[score]`` is truthy.  All original nodes are preserved,
        along with graph, node, and edge attributes.

    See Also
    --------
    threshold_filter : Filter on a numeric score.

    Examples
    --------
    >>> import networkx as nx

    >>> from networkx_backbone import boolean_filter, global_threshold_filter
    >>> G = nx.les_miserables_graph()
    >>> scored = global_threshold_filter(G, threshold=2)
    >>> H = boolean_filter(scored, "global_threshold_keep")
    >>> H.number_of_edges() <= scored.number_of_edges()
    True
    """
    H = _empty_like(G)
    for u, v, key, data in _edge_iter(G):
        if data.get(score):
            _add_edge(H, G, u, v, key, data)
    return H


def consensus_backbone(*backbones):
    """Return the intersection of multiple backbone graphs.

    An edge is in the consensus backbone if and only if it appears in
    **every** input backbone.  Only nodes that are endpoints of consensus
    edges are retained.

    Parameters
    ----------
    *backbones : graph
        Two or more backbone graphs (must share the same node identifiers).

    Returns
    -------
    H : graph
        A new graph of the same type as the first backbone, containing
        only edges present in every input, with the edge, node, and graph
        attributes of the first backbone.  Edges keep the order of the first
        backbone.  Parallel edges of multigraphs are matched by node pair.

    Raises
    ------
    ValueError
        If fewer than 2 backbones are provided.

    Examples
    --------
    >>> import networkx as nx

    >>> from networkx_backbone import (
    ...     consensus_backbone,
    ...     jaccard_backbone,
    ...     cosine_backbone,
    ...     fraction_filter,
    ... )
    >>> G = nx.les_miserables_graph()
    >>> B1 = fraction_filter(jaccard_backbone(G), "jaccard", 0.3, ascending=False)
    >>> B2 = fraction_filter(cosine_backbone(G), "cosine", 0.3, ascending=False)
    >>> H = consensus_backbone(B1, B2)
    >>> H.number_of_edges() <= min(B1.number_of_edges(), B2.number_of_edges())
    True
    """
    if len(backbones) < 2:
        raise ValueError("consensus_backbone requires at least 2 graphs")

    def pair(bb, u, v):
        # frozenset keys work for nodes of mixed, unorderable types.
        return (u, v) if bb.is_directed() else frozenset((u, v))

    others = [{pair(bb, u, v) for u, v in bb.edges()} for bb in backbones[1:]]

    ref = backbones[0]
    H = _empty_like(ref, nodes=False)
    for u, v, key, data in _edge_iter(ref):
        if all(pair(ref, u, v) in es for es in others):
            _add_edge(H, ref, u, v, key, data)
    return H


_COMPLEXITY = {
    "multigraph_to_weighted": {
        "time": "O(m)",
        "space": "O(n + m)",
        "notes": "m counts input edge instances for multigraphs.",
    },
    "threshold_filter": {
        "time": "O(n + m)",
        "space": "O(n + m)",
    },
    "fraction_filter": {
        "time": "O(m log m)",
        "space": "O(n + m)",
        "notes": "Edge mode; node mode is O(n log n).",
    },
    "boolean_filter": {
        "time": "O(n + m)",
        "space": "O(n + m)",
    },
    "consensus_backbone": {
        "time": "O(km)",
        "space": "O(m)",
        "notes": "k=number of input backbones, m=edges per backbone.",
    },
}

append_complexity_docstrings(globals(), _COMPLEXITY)
