"""Evaluation measures for comparing backbones against their original network."""

import inspect

import networkx as nx

from networkx_backbone._docstrings import append_complexity_docstrings

__all__ = [
    "node_fraction",
    "edge_fraction",
    "weight_fraction",
    "reachability",
    "ks_degree",
    "ks_weight",
    "compare_backbones",
]


def node_fraction(original, backbone):
    """Compute the fraction of original nodes that appear in the backbone.

    Only nodes that have at least one edge in the respective graph are
    counted.  Isolated nodes are ignored.

    Parameters
    ----------
    original : graph
        The original NetworkX graph.
    backbone : graph
        The backbone NetworkX graph.

    Returns
    -------
    fraction : float
        Fraction in [0, 1].  Returns 0.0 if *original* has no nodes
        with edges.

    Examples
    --------
    >>> import networkx as nx

    >>> from networkx_backbone import node_fraction
    >>> G = nx.les_miserables_graph()
    >>> H = G.edge_subgraph(list(G.edges())[:100]).copy()
    >>> 0.0 < node_fraction(G, H) <= 1.0
    True
    """
    bb_nodes_with_edges = {n for n in backbone.nodes() if backbone.degree(n) > 0}
    orig_nodes_with_edges = {n for n in original.nodes() if original.degree(n) > 0}
    if len(orig_nodes_with_edges) == 0:
        return 0.0
    return len(bb_nodes_with_edges) / len(orig_nodes_with_edges)


def edge_fraction(original, backbone):
    """Compute the fraction of original edges preserved in the backbone.

    Parameters
    ----------
    original : graph
        The original NetworkX graph.
    backbone : graph
        The backbone NetworkX graph.

    Returns
    -------
    fraction : float
        Fraction in [0, 1].  Returns 0.0 if *original* has no edges.

    Examples
    --------
    >>> import networkx as nx

    >>> from networkx_backbone import edge_fraction
    >>> G = nx.les_miserables_graph()
    >>> H = G.edge_subgraph(list(G.edges())[:100]).copy()
    >>> 0.0 < edge_fraction(G, H) < 1.0
    True
    """
    if original.number_of_edges() == 0:
        return 0.0
    return backbone.number_of_edges() / original.number_of_edges()


def weight_fraction(original, backbone, weight="weight"):
    """Compute the fraction of total edge weight preserved in the backbone.

    Parameters
    ----------
    original : graph
        The original NetworkX graph.
    backbone : graph
        The backbone NetworkX graph.
    weight : string, optional (default="weight")
        Edge attribute key used as the weight.

    Returns
    -------
    fraction : float
        Fraction in [0, 1].  Returns 0.0 if *original* has zero total
        weight.

    Examples
    --------
    >>> import networkx as nx

    >>> from networkx_backbone import (
    ...     weight_fraction,
    ...     global_threshold_filter,
    ...     boolean_filter,
    ... )
    >>> G = nx.les_miserables_graph()
    >>> scored = global_threshold_filter(G, threshold=2)
    >>> H = boolean_filter(scored, "global_threshold_keep")
    >>> 0.0 < weight_fraction(G, H) <= 1.0
    True
    """
    total_orig = sum(d.get(weight, 0) for _, _, d in original.edges(data=True))
    if total_orig == 0:
        return 0.0
    total_bb = sum(d.get(weight, 0) for _, _, d in backbone.edges(data=True))
    return total_bb / total_orig


def reachability(G):
    """Compute the fraction of ordered node pairs joined by a path.

    For a connected undirected graph (or a strongly connected directed graph)
    this is 1.0.  For a graph with all isolated nodes this is 0.0.  For a
    directed graph, a pair *(i, j)* counts only if *j* can be reached from
    *i* along edge directions.  Graphs with fewer than two nodes have no
    pairs; their reachability is defined as 1.0.

    Parameters
    ----------
    G : graph
        A NetworkX graph.

    Returns
    -------
    fraction : float
        Fraction in [0, 1] of ordered node pairs *(i, j)* for which a
        path exists.

    Examples
    --------
    >>> import networkx as nx

    >>> from networkx_backbone import reachability
    >>> G = nx.les_miserables_graph()
    >>> reachability(G)
    1.0
    >>> H = nx.Graph()
    >>> H.add_nodes_from(G.nodes())
    >>> reachability(H)
    0.0
    """
    n = G.number_of_nodes()
    if n <= 1:
        return 1.0
    total_pairs = n * (n - 1)

    if not G.is_directed():
        reachable = sum(len(c) * (len(c) - 1) for c in nx.connected_components(G))
        return reachable / total_pairs

    # Every node of a strongly connected component reaches the rest of its
    # component and every node of each component downstream of it.
    C = nx.condensation(G)
    size = {c: len(C.nodes[c]["members"]) for c in C}
    reachable = 0
    for c in C:
        downstream = sum(size[d] for d in nx.descendants(C, c))
        reachable += size[c] * (size[c] - 1 + downstream)
    return reachable / total_pairs


def ks_degree(original, backbone):
    """Compute the Kolmogorov-Smirnov statistic between degree distributions.

    Parameters
    ----------
    original : graph
        The original NetworkX graph.
    backbone : graph
        The backbone NetworkX graph.

    Returns
    -------
    statistic : float
        KS statistic in [0, 1].

    Examples
    --------
    >>> import networkx as nx

    >>> from networkx_backbone import ks_degree, global_threshold_filter, boolean_filter
    >>> G = nx.les_miserables_graph()
    >>> H = boolean_filter(global_threshold_filter(G, threshold=2), "global_threshold_keep")
    >>> 0.0 <= ks_degree(G, H) <= 1.0
    True
    """
    import numpy as np
    from scipy import stats as sp_stats

    orig_deg = np.array([d for _, d in original.degree()])
    bb_deg = np.array([d for _, d in backbone.degree()])
    if len(orig_deg) == 0 or len(bb_deg) == 0:
        return 1.0
    stat, _ = sp_stats.ks_2samp(orig_deg, bb_deg)
    return float(stat)


def ks_weight(original, backbone, weight="weight"):
    """Compute the Kolmogorov-Smirnov statistic between weight distributions.

    Parameters
    ----------
    original : graph
        The original NetworkX graph.
    backbone : graph
        The backbone NetworkX graph.
    weight : string, optional (default="weight")
        Edge attribute key used as the weight.

    Returns
    -------
    statistic : float
        KS statistic in [0, 1].

    Examples
    --------
    >>> import networkx as nx

    >>> from networkx_backbone import ks_weight, global_threshold_filter, boolean_filter
    >>> G = nx.les_miserables_graph()
    >>> H = boolean_filter(global_threshold_filter(G, threshold=2), "global_threshold_keep")
    >>> 0.0 <= ks_weight(G, H) <= 1.0
    True
    """
    import numpy as np
    from scipy import stats as sp_stats

    orig_w = np.array([d.get(weight, 0) for _, _, d in original.edges(data=True)])
    bb_w = np.array([d.get(weight, 0) for _, _, d in backbone.edges(data=True)])
    if len(orig_w) == 0 or len(bb_w) == 0:
        return 1.0
    stat, _ = sp_stats.ks_2samp(orig_w, bb_w)
    return float(stat)


def compare_backbones(original, backbones, measures=None, weight="weight"):
    """Compare multiple backbones on a set of evaluation measures.

    Parameters
    ----------
    original : graph
        The original NetworkX graph.
    backbones : dict
        Mapping ``{name: backbone_graph}``.
    measures : list of callable, optional
        Each callable has signature ``(original, backbone)`` and returns a
        float.  Defaults to ``[node_fraction, edge_fraction,
        weight_fraction]``.
    weight : string, optional (default="weight")
        Weight attribute, forwarded to every measure that accepts a
        ``weight`` keyword (such as :func:`weight_fraction` and
        :func:`ks_weight`).

    Returns
    -------
    results : dict
        ``{backbone_name: {measure_name: value}}``.

    Examples
    --------
    >>> import networkx as nx

    >>> from networkx_backbone import (
    ...     compare_backbones,
    ...     edge_fraction,
    ...     global_threshold_filter,
    ...     boolean_filter,
    ... )
    >>> G = nx.les_miserables_graph()
    >>> H = boolean_filter(global_threshold_filter(G, threshold=2), "global_threshold_keep")
    >>> results = compare_backbones(G, {"threshold": H}, measures=[edge_fraction])
    >>> "edge_fraction" in results["threshold"]
    True
    """
    if measures is None:
        measures = [node_fraction, edge_fraction, weight_fraction]

    takes_weight = {m: _accepts_weight(m) for m in measures}
    results = {}
    for name, bb in backbones.items():
        results[name] = {}
        for m in measures:
            kwargs = {"weight": weight} if takes_weight[m] else {}
            results[name][getattr(m, "__name__", repr(m))] = m(original, bb, **kwargs)
    return results


def _accepts_weight(func):
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False
    return "weight" in params or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    )


_COMPLEXITY = {
    "node_fraction": {
        "time": "O(n)",
        "space": "O(n)",
    },
    "edge_fraction": {
        "time": "O(1)",
        "space": "O(1)",
    },
    "weight_fraction": {
        "time": "O(m)",
        "space": "O(1)",
    },
    "reachability": {
        "time": "O(n + m)",
        "space": "O(n)",
    },
    "ks_degree": {
        "time": "O(n)",
        "space": "O(n)",
    },
    "ks_weight": {
        "time": "O(m)",
        "space": "O(m)",
    },
    "compare_backbones": {
        "time": "O(b * C)",
        "space": "O(b * q)",
        "notes": "b=backbones, q=measures, C=cost per measure evaluation.",
    },
}

append_complexity_docstrings(globals(), _COMPLEXITY)
