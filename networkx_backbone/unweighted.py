"""
Unweighted network backbone methods.

These methods sparsify unweighted graphs by scoring edges and filtering
based on local topology.

Methods
-------
sparsify
    Generic sparsification framework (score -> normalise -> filter -> connect).
lspar
    Local Sparsification (Satuluri et al. 2011) -- Jaccard-based.
local_degree
    Local Degree (Lindner et al. 2015) -- keep edges to high-degree neighbours.
"""

import math
from bisect import bisect_right

import networkx as nx
from networkx.utils import not_implemented_for, py_random_state

from networkx_backbone._docstrings import append_complexity_docstrings

__all__ = ["sparsify", "lspar", "local_degree"]

_ESCORES = ("jaccard", "degree", "triangles", "quadrangles", "random")
_NORMALIZE = ("rank", None)
_FILTERS = ("degree", "threshold")


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@py_random_state("seed")
def sparsify(
    G, escore="jaccard", normalize="rank", filter="degree", s=0.5, umst=False, seed=None
):
    r"""Sparsify an unweighted graph using a score-normalise-filter pipeline.

    Follows the four-step pipeline from Neal (2022) [1]_:

    1. **Score**: every node *u* scores each of its incident edges
       ``(u, v)``.  Some scores are symmetric (Jaccard, triangles,
       quadrangles, random); ``"degree"`` is asymmetric: *u* scores the
       edge by the degree of the *neighbour* *v*.
    2. **Normalise**: optionally rank-transform the scores within each
       node's neighbourhood.
    3. **Filter**: retain the top ``max(1, ceil(d_u**s))`` edges of every
       node *u* (``filter="degree"``), or every edge whose score is at
       least *s* (``filter="threshold"``).  An edge is kept if *either*
       endpoint retains it.
    4. **Connect**: optionally add the edges of a spanning forest so the
       backbone has as many connected components as *G*.

    Parameters
    ----------
    G : graph
        An undirected NetworkX graph.  Edge attributes are ignored.
    escore : string, optional (default="jaccard")
        Edge scoring method.  One of ``"jaccard"`` (Jaccard similarity of
        the endpoint neighbourhoods), ``"degree"`` (degree of the
        neighbour), ``"triangles"`` (number of common neighbours),
        ``"quadrangles"`` (number of 4-cycles through the edge) or
        ``"random"`` (uniform random score in [0, 1)).
    normalize : {"rank", None}, optional (default="rank")
        ``"rank"`` replaces each node's view of an edge score by its
        normalised rank among that node's edges, ``r / d_u``, where *r* is
        the number of the node's edges scoring less than or equal to the
        edge (so the best edge has value 1 and tied edges share a value).
        ``None`` keeps the raw scores.  With ``filter="degree"`` both
        choices give the same backbone because ranking is monotone.
    filter : {"degree", "threshold"}, optional (default="degree")
        Filtering method (see *Notes*).
    s : float, optional (default=0.5)
        Sparsification parameter: the exponent for ``filter="degree"``
        (typically in [0, 1]; 0 is sparsest, 1 keeps every edge) or the
        score threshold for ``filter="threshold"``.
    umst : bool, optional (default=False)
        If ``True`` and the retained edges split a connected component
        of *G*, add the edges of a minimum spanning forest of *G* computed
        on the edge scores (preferring the most important edges) so that
        connectivity is restored.  This is a single spanning forest, not
        the union of all minimum spanning trees.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state, used only when
        ``escore="random"``.
        See :ref:`Randomness<networkx:randomness>`.

    Returns
    -------
    H : graph
        A copy of *G* with edge attributes ``"sparsify_score"`` (float, see
        *Notes*) and boolean ``"sparsify_keep"``.

    Raises
    ------
    ValueError
        If *escore*, *normalize* or *filter* is not one of the documented
        values.
    NetworkXNotImplemented
        If *G* is directed or a multigraph.  Collapse a multigraph first
        with :func:`~networkx_backbone.multigraph_to_weighted`.

    Notes
    -----
    Ranks are *competition ranks*: the rank of edge *e* at node *u* is
    ``1 + #{edges of u scoring strictly better than e}``, so tied edges
    share a rank and are retained or dropped together.  Self-loops are
    ignored when scoring and ranking and are never retained; *d_u* is the
    number of distinct neighbours of *u*.

    With ``filter="degree"`` node *u* keeps the edges whose rank *r*
    satisfies ``r <= max(1, ceil(d_u**s))``, and ``"sparsify_score"`` is
    the *critical exponent* of the edge,

    .. math::

        s^*(e) = \min_{u \in e} \frac{\log(r_u(e) - 1)}{\log d_u},

    with :math:`s^* = -\infty` for an edge that is top-ranked at one of
    its endpoints.  The edge is retained exactly when ``s > s*``, so
    ``threshold_filter(H, "sparsify_score", s2, mode="below")`` gives the
    backbone for any other exponent ``s2`` without re-scoring.  Smaller
    scores mean more important edges.

    With ``filter="threshold"``, ``"sparsify_score"`` is the larger of the
    two endpoints' (normalised) scores and the edge is retained when it is
    at least *s*.

    References
    ----------
    .. [1] Neal, Z. P. (2022). backbone: An R package to extract network
       backbones. *PLOS ONE*, 17(5), e0269137.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import boolean_filter, sparsify
    >>> G_weighted = nx.les_miserables_graph()
    >>> G = nx.Graph()
    >>> G.add_nodes_from(G_weighted.nodes(data=True))
    >>> G.add_edges_from(G_weighted.edges())
    >>> H = sparsify(G, s=0.5)
    >>> backbone = boolean_filter(H, "sparsify_keep")
    >>> backbone.number_of_edges() <= H.number_of_edges()
    True
    """
    return _sparsify(G, escore, normalize, filter, s, umst, seed, rounding="ceil")


@not_implemented_for("directed")
@not_implemented_for("multigraph")
def lspar(G, s=0.5):
    r"""Compute the Local Sparsification (L-Spar) backbone.

    Every node *u* ranks its incident edges by the Jaccard similarity of
    the endpoint neighbourhoods and keeps its top ``ceil(d_u**s)`` edges;
    an edge is retained if either endpoint keeps it [1]_.  Equivalent to
    ``sparsify(G, escore="jaccard", normalize="rank", filter="degree", s=s)``.

    Parameters
    ----------
    G : graph
        An undirected NetworkX graph.
    s : float, optional (default=0.5)
        Sparsification exponent (0 = sparsest, 1 = keep all).

    Returns
    -------
    H : graph
        A copy of *G* with ``"sparsify_score"`` (the critical exponent,
        see :func:`sparsify`) and boolean ``"sparsify_keep"``.

    Raises
    ------
    NetworkXNotImplemented
        If *G* is directed or a multigraph.

    Notes
    -----
    Edges tied in Jaccard similarity at the cut-off are all retained
    (competition ranking), which makes the result independent of node
    order.  ``threshold_filter(H, "sparsify_score", s2, mode="below")``
    reproduces ``lspar(G, s=s2)`` for any exponent ``s2``.

    References
    ----------
    .. [1] Satuluri, V., Parthasarathy, S., & Ruan, Y. (2011). Local
       graph sparsification for scalable clustering. *ACM SIGMOD*, 721-732.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import boolean_filter, lspar
    >>> G_weighted = nx.les_miserables_graph()
    >>> G = nx.Graph()
    >>> G.add_nodes_from(G_weighted.nodes(data=True))
    >>> G.add_edges_from(G_weighted.edges())
    >>> H = lspar(G, s=0.5)
    >>> backbone = boolean_filter(H, "sparsify_keep")
    >>> backbone.number_of_edges() <= H.number_of_edges()
    True
    """
    return _sparsify(G, "jaccard", "rank", "degree", s, False, None, rounding="ceil")


@not_implemented_for("directed")
@not_implemented_for("multigraph")
def local_degree(G, s=0.3):
    r"""Compute the Local Degree backbone.

    Every node *u* ranks its neighbours by their degree (highest first)
    and keeps the edges to its top ``floor(d_u**s)`` neighbours; an edge is
    retained if either endpoint keeps it [1]_, [2]_.  Edges to hubs are
    therefore preferred, which preserves the network's hub structure.

    Parameters
    ----------
    G : graph
        An undirected NetworkX graph.
    s : float, optional (default=0.3)
        Sparsification exponent (0 = sparsest, 1 = keep all).

    Returns
    -------
    H : graph
        A copy of *G* with ``"sparsify_score"`` and boolean
        ``"sparsify_keep"``.

    Raises
    ------
    NetworkXNotImplemented
        If *G* is directed or a multigraph.

    Notes
    -----
    Neighbours of equal degree share a rank (competition ranking), so
    ties at the cut-off are all retained.  Each node always keeps at
    least its top-ranked edge(s).  ``"sparsify_score"`` is the critical
    exponent

    .. math::

        s^*(e) = \min_{u \in e} \frac{\log r_u(e)}{\log d_u}

    (:math:`-\infty` for edges top-ranked at an endpoint), and the edge is
    retained exactly when ``s >= s*``.  Smaller scores mean more important
    edges.

    References
    ----------
    .. [1] Lindner, G., Staudt, C. L., Hamann, M., Meyerhenke, H., &
       Wagner, D. (2015). Structure-preserving sparsification of social
       networks. *Proceedings of the 2015 IEEE/ACM International
       Conference on Advances in Social Networks Analysis and Mining
       (ASONAM)*, 448-454.
    .. [2] Hamann, M., Lindner, G., Meyerhenke, H., Staudt, C. L., &
       Wagner, D. (2016). Structure-preserving sparsification methods for
       social networks. *Social Network Analysis and Mining*, 6(1), 22.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import boolean_filter, local_degree
    >>> G = nx.star_graph(4)
    >>> G.add_edges_from([(4, 5), (4, 6)])
    >>> H = local_degree(G, s=0.0)
    >>> sorted(e for e in H.edges() if H.edges[e]["sparsify_keep"])
    [(0, 1), (0, 2), (0, 3), (0, 4), (4, 5), (4, 6)]
    >>> H[0][4]["sparsify_keep"]  # the hub keeps the edge to its best-connected neighbour
    True
    """
    return _sparsify(G, "degree", "rank", "degree", s, False, None, rounding="floor")


# =====================================================================
# Internal helpers
# =====================================================================


def _sparsify(G, escore, normalize, filter, s, umst, seed, rounding):
    """Shared implementation of :func:`sparsify`, :func:`lspar`, :func:`local_degree`."""
    if escore not in _ESCORES:
        raise ValueError(
            f"Unknown edge scoring method: {escore!r}; expected one of {_ESCORES}"
        )
    if normalize not in _NORMALIZE:
        raise ValueError(f"Unknown normalize: {normalize!r}; expected 'rank' or None")
    if filter not in _FILTERS:
        raise ValueError(f"Unknown filter: {filter!r}; expected one of {_FILTERS}")

    nbrs = {u: set(G[u]) - {u} for u in G}
    views = _score_views(G, nbrs, escore, seed)

    if filter == "degree":
        scores = _critical_exponents(views, rounding)
        if rounding == "floor":
            kept = {e for e, sc in scores.items() if sc <= s}
        else:
            kept = {e for e, sc in scores.items() if sc < s}
        badness = scores
    else:
        if normalize == "rank":
            views = _rank_normalise(views)
        scores = {}
        for u, view in views.items():
            for v, sc in view.items():
                e = frozenset((u, v))
                scores[e] = max(scores.get(e, sc), sc)
        kept = {e for e, sc in scores.items() if sc >= s}
        badness = {e: -sc for e, sc in scores.items()}

    if umst:
        selected = nx.Graph()
        selected.add_nodes_from(G)
        selected.add_edges_from(tuple(e) for e in kept)
        n_components = nx.number_connected_components(G)
        if nx.number_connected_components(selected) > n_components:
            scored = nx.Graph()
            scored.add_nodes_from(G)
            for e, b in badness.items():
                u, v = tuple(e)
                scored.add_edge(u, v, b=b)
            for u, v in nx.minimum_spanning_edges(scored, weight="b", data=False):
                kept.add(frozenset((u, v)))

    self_loop_score = math.inf if filter == "degree" else -math.inf
    H = G.copy()
    for u, v, data in H.edges(data=True):
        e = frozenset((u, v))
        if u == v:
            data["sparsify_score"] = self_loop_score
            data["sparsify_keep"] = False
        else:
            data["sparsify_score"] = float(scores[e])
            data["sparsify_keep"] = e in kept
    return H


def _score_views(G, nbrs, method, seed):
    """Return ``{u: {v: score of edge (u, v) as seen from u}}`` (no self-loops)."""
    views = {u: {} for u in G}

    if method == "random":
        for u, v in G.edges():
            if u != v:
                views[u][v] = views[v][u] = seed.random()
        return views

    for u, v in G.edges():
        if u == v:
            continue
        nu, nv = nbrs[u], nbrs[v]
        if method == "jaccard":
            union = len(nu | nv)
            sc = len(nu & nv) / union if union else 0.0
            views[u][v] = views[v][u] = sc
        elif method == "degree":
            views[u][v] = len(nv)
            views[v][u] = len(nu)
        elif method == "triangles":
            views[u][v] = views[v][u] = len(nu & nv)
        else:  # quadrangles: 4-cycles u - x - y - v - u
            nu_x = nu - {v}
            nv_y = nv - {u}
            count = sum(len((nbrs[x] - {u}) & nv_y) for x in nu_x)
            views[u][v] = views[v][u] = count
    return views


def _rank_normalise(views):
    """Replace each node's scores by normalised ranks ``#{<= score} / d``."""
    normalised = {}
    for u, view in views.items():
        ordered = sorted(view.values())
        d = len(ordered)
        normalised[u] = {v: bisect_right(ordered, sc) / d for v, sc in view.items()}
    return normalised


def _critical_exponents(views, rounding):
    """Per-edge critical exponent of the per-node top-k ("degree") filter.

    Node *u* with ``d`` distinct neighbours keeps an edge of competition
    rank ``r`` iff ``r <= max(1, ceil(d**s))`` (``rounding="ceil"``), i.e.
    iff ``s > log(r - 1) / log(d)``; or iff ``r <= max(1, floor(d**s))``
    (``rounding="floor"``), i.e. iff ``s >= log(r) / log(d)``.  Top-ranked
    edges get ``-inf``.  The edge score is the minimum over both endpoints.
    """
    exponents = {}
    for u, view in views.items():
        d = len(view)
        ordered = sorted(view.values())
        for v, sc in view.items():
            # competition rank: 1 + number of strictly better edges
            r = 1 + d - bisect_right(ordered, sc)
            if r == 1:
                e_u = -math.inf
            elif rounding == "ceil":
                e_u = math.log(r - 1) / math.log(d)
            else:
                e_u = math.log(r) / math.log(d)
            e = frozenset((u, v))
            exponents[e] = min(exponents.get(e, e_u), e_u)
    return exponents


_COMPLEXITY = {
    "sparsify": {
        "time": "O(mn)",
        "space": "O(n + m)",
        "notes": "Worst-case; depends on escore/filter choices.",
    },
    "lspar": {
        "time": "O(mn)",
        "space": "O(n + m)",
        "notes": "Jaccard scoring plus per-node sorting.",
    },
    "local_degree": {
        "time": "O(m log n)",
        "space": "O(n + m)",
        "notes": "Per-node sorting of neighbour degrees.",
    },
}

append_complexity_docstrings(globals(), _COMPLEXITY)
