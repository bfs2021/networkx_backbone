"""
Proximity-based edge scoring methods.

These methods score each existing edge by the topological proximity
(similarity) of its endpoints.  They are useful for identifying
structurally embedded edges (high proximity) versus bridge edges
(low proximity) and can feed into any of the filtering utilities
in :mod:`networkx_backbone.filters`.

All local methods use the same conventions: the neighbourhood
:math:`N(u)` of a node excludes the node itself (self-loops are ignored,
so an endpoint is never counted as a common neighbour of an edge), and
:math:`k_u = |N(u)|`.  For directed graphs neighbourhoods are successor
sets and :math:`k_u` is the out-degree.  Multigraphs are not supported;
collapse them first with :func:`~networkx_backbone.multigraph_to_weighted`.

Local methods
-------------
neighborhood_overlap
    Raw common-neighbor count.
jaccard_backbone
    Jaccard coefficient of endpoint neighborhoods.
dice_backbone
    Dice / Sorensen coefficient of endpoint neighborhoods.
cosine_backbone
    Cosine / Salton index of endpoint neighborhoods.
hub_promoted_index
    Common neighbors normalised by the smaller degree.
hub_depressed_index
    Common neighbors normalised by the larger degree.
lhn_local_index
    Leicht-Holme-Newman local similarity index.
preferential_attachment_score
    Product of endpoint degrees.
adamic_adar_index
    Adamic-Adar index (inverse-log-degree-weighted common neighbors).
resource_allocation_index
    Resource-allocation index (inverse-degree-weighted common neighbors).

Quasi-local methods
-------------------
graph_distance_proximity
    Reciprocal shortest-path distance.
local_path_index
    Local path index (second- and third-order path counts).

References
----------
Lu, L. & Zhou, T. (2011). "Link prediction in complex networks:
A survey." *Physica A*, 390(6), 1150-1170.

Liben-Nowell, D. & Kleinberg, J. (2007). "The link-prediction problem
for social networks." *JASIST*, 58(7), 1019-1031.
"""

import math

import networkx as nx
from networkx.utils import not_implemented_for

from networkx_backbone._docstrings import append_complexity_docstrings

__all__ = [
    "neighborhood_overlap",
    "jaccard_backbone",
    "dice_backbone",
    "cosine_backbone",
    "hub_promoted_index",
    "hub_depressed_index",
    "lhn_local_index",
    "preferential_attachment_score",
    "adamic_adar_index",
    "resource_allocation_index",
    "graph_distance_proximity",
    "local_path_index",
]


# =====================================================================
# Helper
# =====================================================================


def _neighbor_sets(G):
    """Return ``{node: neighbours}``, excluding the node itself.

    For directed graphs the neighbours are the successors.  Excluding the
    node itself means self-loops never create a common neighbour.
    """
    return {v: set(G[v]) - {v} for v in G}


def _degrees(nbrs):
    """Degree (out-degree for directed graphs) consistent with *nbrs*."""
    return {v: len(s) for v, s in nbrs.items()}


# =====================================================================
# Local methods
# =====================================================================


@not_implemented_for("multigraph")
def neighborhood_overlap(G):
    """Score each edge by the raw neighborhood overlap of its endpoints.

    For each edge (u, v), the overlap is the number of common neighbors:
    ``|N(u) & N(v)|``.  The score is stored as the ``"overlap"`` edge
    attribute.

    Parameters
    ----------
    G : networkx.Graph or networkx.DiGraph
        Input graph. For directed graphs, neighborhoods are defined by
        successors (out-neighbors); see the module notes.

    Returns
    -------
    H : graph
        Copy of *G* with the ``"overlap"`` edge attribute added.

    Raises
    ------
    NetworkXNotImplemented
        If *G* is a multigraph.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import fraction_filter, neighborhood_overlap
    >>> G = nx.les_miserables_graph()
    >>> H = neighborhood_overlap(G)
    >>> backbone = fraction_filter(H, "overlap", 0.3, ascending=False)
    >>> backbone.number_of_edges() <= H.number_of_edges()
    True
    """
    H = G.copy()
    nbrs = _neighbor_sets(G)
    for u, v in H.edges():
        H[u][v]["overlap"] = len(nbrs[u] & nbrs[v])
    return H


@not_implemented_for("multigraph")
def jaccard_backbone(G):
    r"""Score each edge by the Jaccard similarity of its endpoint neighborhoods.

    For each edge (u, v):

    .. math::

        J_{uv} = \frac{|N(u) \cap N(v)|}{|N(u) \cup N(v)|}

    The score is stored as the ``"jaccard"`` edge attribute.

    Parameters
    ----------
    G : networkx.Graph or networkx.DiGraph
        Input graph.

    Returns
    -------
    H : graph
        Copy of *G* with the ``"jaccard"`` edge attribute added.

    Raises
    ------
    NetworkXNotImplemented
        If *G* is a multigraph.

    References
    ----------
    .. [1] Jaccard, P. (1901). Distribution de la flore alpine dans le
       bassin des Dranses et dans quelques regions voisines. *Bulletin
       de la Societe Vaudoise des Sciences Naturelles*, 37, 241-272.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import fraction_filter, jaccard_backbone
    >>> G = nx.les_miserables_graph()
    >>> H = jaccard_backbone(G)
    >>> backbone = fraction_filter(H, "jaccard", 0.3, ascending=False)
    >>> backbone.number_of_edges() <= H.number_of_edges()
    True
    """
    H = G.copy()
    nbrs = _neighbor_sets(G)
    for u, v in H.edges():
        nu = nbrs[u]
        nv = nbrs[v]
        intersection = len(nu & nv)
        union = len(nu | nv)
        H[u][v]["jaccard"] = intersection / union if union > 0 else 0.0
    return H


@not_implemented_for("multigraph")
def dice_backbone(G):
    r"""Score each edge by the Dice similarity of its endpoint neighborhoods.

    For each edge (u, v):

    .. math::

        D_{uv} = \frac{2 \, |N(u) \cap N(v)|}{k_u + k_v}

    The score is stored as the ``"dice"`` edge attribute.

    Parameters
    ----------
    G : networkx.Graph or networkx.DiGraph
        Input graph.

    Returns
    -------
    H : graph
        Copy of *G* with the ``"dice"`` edge attribute added.

    Raises
    ------
    NetworkXNotImplemented
        If *G* is a multigraph.

    References
    ----------
    .. [1] Dice, L. R. (1945). Measures of the amount of ecologic
       association between species. *Ecology*, 26(3), 297-302.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import dice_backbone, fraction_filter
    >>> G = nx.les_miserables_graph()
    >>> H = dice_backbone(G)
    >>> backbone = fraction_filter(H, "dice", 0.3, ascending=False)
    >>> backbone.number_of_edges() <= H.number_of_edges()
    True
    """
    H = G.copy()
    nbrs = _neighbor_sets(G)
    deg = _degrees(nbrs)
    for u, v in H.edges():
        intersection = len(nbrs[u] & nbrs[v])
        denom = deg[u] + deg[v]
        H[u][v]["dice"] = (2.0 * intersection) / denom if denom > 0 else 0.0
    return H


@not_implemented_for("multigraph")
def cosine_backbone(G):
    r"""Score each edge by the cosine similarity of its endpoint neighborhoods.

    For each edge (u, v):

    .. math::

        C_{uv} = \frac{|N(u) \cap N(v)|}{\sqrt{k_u \, k_v}}

    The score is stored as the ``"cosine"`` edge attribute.

    Parameters
    ----------
    G : networkx.Graph or networkx.DiGraph
        Input graph.

    Returns
    -------
    H : graph
        Copy of *G* with the ``"cosine"`` edge attribute added.

    Raises
    ------
    NetworkXNotImplemented
        If *G* is a multigraph.

    References
    ----------
    .. [1] Salton, G. & McGill, M. J. (1983). *Introduction to Modern
       Information Retrieval*. McGraw-Hill.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import cosine_backbone, fraction_filter
    >>> G = nx.les_miserables_graph()
    >>> H = cosine_backbone(G)
    >>> backbone = fraction_filter(H, "cosine", 0.3, ascending=False)
    >>> backbone.number_of_edges() <= H.number_of_edges()
    True
    """
    H = G.copy()
    nbrs = _neighbor_sets(G)
    deg = _degrees(nbrs)
    for u, v in H.edges():
        intersection = len(nbrs[u] & nbrs[v])
        denom = math.sqrt(deg[u] * deg[v])
        H[u][v]["cosine"] = intersection / denom if denom > 0 else 0.0
    return H


@not_implemented_for("multigraph")
def hub_promoted_index(G):
    r"""Score each edge by the Hub Promoted Index of its endpoint neighborhoods.

    For each edge (u, v):

    .. math::

        \mathrm{HPI}_{uv} = \frac{|N(u) \cap N(v)|}{\min(k_u,\, k_v)}

    The score is stored as the ``"hpi"`` edge attribute.

    Parameters
    ----------
    G : networkx.Graph or networkx.DiGraph
        Input graph.

    Returns
    -------
    H : graph
        Copy of *G* with the ``"hpi"`` edge attribute added.

    Raises
    ------
    NetworkXNotImplemented
        If *G* is a multigraph.

    References
    ----------
    .. [1] Ravasz, E., Somera, A. L., Mongru, D. A., Oltvai, Z. N. &
       Barabasi, A.-L. (2002). Hierarchical organization of modularity
       in metabolic networks. *Science*, 297(5586), 1551-1555.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import fraction_filter, hub_promoted_index
    >>> G = nx.les_miserables_graph()
    >>> H = hub_promoted_index(G)
    >>> backbone = fraction_filter(H, "hpi", 0.3, ascending=False)
    >>> backbone.number_of_edges() <= H.number_of_edges()
    True
    """
    H = G.copy()
    nbrs = _neighbor_sets(G)
    deg = _degrees(nbrs)
    for u, v in H.edges():
        intersection = len(nbrs[u] & nbrs[v])
        denom = min(deg[u], deg[v])
        H[u][v]["hpi"] = intersection / denom if denom > 0 else 0.0
    return H


@not_implemented_for("multigraph")
def hub_depressed_index(G):
    r"""Score each edge by the Hub Depressed Index of its endpoint neighborhoods.

    For each edge (u, v):

    .. math::

        \mathrm{HDI}_{uv} = \frac{|N(u) \cap N(v)|}{\max(k_u,\, k_v)}

    The score is stored as the ``"hdi"`` edge attribute.

    Parameters
    ----------
    G : networkx.Graph or networkx.DiGraph
        Input graph.

    Returns
    -------
    H : graph
        Copy of *G* with the ``"hdi"`` edge attribute added.

    Raises
    ------
    NetworkXNotImplemented
        If *G* is a multigraph.

    References
    ----------
    .. [1] Ravasz, E., Somera, A. L., Mongru, D. A., Oltvai, Z. N. &
       Barabasi, A.-L. (2002). Hierarchical organization of modularity
       in metabolic networks. *Science*, 297(5586), 1551-1555.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import fraction_filter, hub_depressed_index
    >>> G = nx.les_miserables_graph()
    >>> H = hub_depressed_index(G)
    >>> backbone = fraction_filter(H, "hdi", 0.3, ascending=False)
    >>> backbone.number_of_edges() <= H.number_of_edges()
    True
    """
    H = G.copy()
    nbrs = _neighbor_sets(G)
    deg = _degrees(nbrs)
    for u, v in H.edges():
        intersection = len(nbrs[u] & nbrs[v])
        denom = max(deg[u], deg[v])
        H[u][v]["hdi"] = intersection / denom if denom > 0 else 0.0
    return H


@not_implemented_for("multigraph")
def lhn_local_index(G):
    r"""Score each edge by the Leicht-Holme-Newman local similarity index.

    For each edge (u, v):

    .. math::

        \mathrm{LHN}_{uv} = \frac{|N(u) \cap N(v)|}{k_u \cdot k_v}

    The score is stored as the ``"lhn_local"`` edge attribute.

    Parameters
    ----------
    G : networkx.Graph or networkx.DiGraph
        Input graph.

    Returns
    -------
    H : graph
        Copy of *G* with the ``"lhn_local"`` edge attribute added.

    Raises
    ------
    NetworkXNotImplemented
        If *G* is a multigraph.

    References
    ----------
    .. [1] Leicht, E. A., Holme, P. & Newman, M. E. J. (2006). Vertex
       similarity in networks. *Physical Review E*, 73(2), 026120.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import fraction_filter, lhn_local_index
    >>> G = nx.les_miserables_graph()
    >>> H = lhn_local_index(G)
    >>> backbone = fraction_filter(H, "lhn_local", 0.3, ascending=False)
    >>> backbone.number_of_edges() <= H.number_of_edges()
    True
    """
    H = G.copy()
    nbrs = _neighbor_sets(G)
    deg = _degrees(nbrs)
    for u, v in H.edges():
        intersection = len(nbrs[u] & nbrs[v])
        denom = deg[u] * deg[v]
        H[u][v]["lhn_local"] = intersection / denom if denom > 0 else 0.0
    return H


@not_implemented_for("multigraph")
def preferential_attachment_score(G):
    r"""Score each edge by the preferential attachment index.

    For each edge (u, v):

    .. math::

        \mathrm{PA}_{uv} = k_u \cdot k_v

    The score is stored as the ``"pa"`` edge attribute.

    Parameters
    ----------
    G : networkx.Graph or networkx.DiGraph
        Input graph.

    Returns
    -------
    H : graph
        Copy of *G* with the ``"pa"`` edge attribute added.

    Raises
    ------
    NetworkXNotImplemented
        If *G* is a multigraph.

    References
    ----------
    .. [1] Barabasi, A.-L. & Albert, R. (1999). Emergence of scaling in
       random networks. *Science*, 286(5439), 509-512.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import fraction_filter, preferential_attachment_score
    >>> G = nx.les_miserables_graph()
    >>> H = preferential_attachment_score(G)
    >>> backbone = fraction_filter(H, "pa", 0.3, ascending=False)
    >>> backbone.number_of_edges() <= H.number_of_edges()
    True
    """
    H = G.copy()
    deg = _degrees(_neighbor_sets(G))
    for u, v in H.edges():
        H[u][v]["pa"] = deg[u] * deg[v]
    return H


@not_implemented_for("multigraph")
def adamic_adar_index(G):
    r"""Score each edge by the Adamic-Adar index.

    For each edge (u, v):

    .. math::

        \mathrm{AA}_{uv} = \sum_{z \,\in\, N(u) \cap N(v)}
                           \frac{1}{\log k_z}

    The score is stored as the ``"adamic_adar"`` edge attribute.

    Parameters
    ----------
    G : networkx.Graph or networkx.DiGraph
        Input graph.

    Returns
    -------
    H : graph
        Copy of *G* with the ``"adamic_adar"`` edge attribute added.

    Raises
    ------
    NetworkXNotImplemented
        If *G* is a multigraph.

    References
    ----------
    .. [1] Adamic, L. A. & Adar, E. (2003). Friends and neighbors on the
       Web. *Social Networks*, 25(3), 211-230.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import adamic_adar_index, fraction_filter
    >>> G = nx.les_miserables_graph()
    >>> H = adamic_adar_index(G)
    >>> backbone = fraction_filter(H, "adamic_adar", 0.3, ascending=False)
    >>> backbone.number_of_edges() <= H.number_of_edges()
    True
    """
    H = G.copy()
    nbrs = _neighbor_sets(G)
    deg = _degrees(nbrs)
    for u, v in H.edges():
        common = nbrs[u] & nbrs[v]
        score = sum(1.0 / math.log(deg[z]) for z in common if deg[z] > 1)
        H[u][v]["adamic_adar"] = score
    return H


@not_implemented_for("multigraph")
def resource_allocation_index(G):
    r"""Score each edge by the resource-allocation index.

    For each edge (u, v):

    .. math::

        \mathrm{RA}_{uv} = \sum_{z \,\in\, N(u) \cap N(v)}
                           \frac{1}{k_z}

    The score is stored as the ``"resource_allocation"`` edge attribute.

    Parameters
    ----------
    G : networkx.Graph or networkx.DiGraph
        Input graph.

    Returns
    -------
    H : graph
        Copy of *G* with the ``"resource_allocation"`` edge attribute
        added.

    Raises
    ------
    NetworkXNotImplemented
        If *G* is a multigraph.

    References
    ----------
    .. [1] Zhou, T., Lu, L. & Zhang, Y.-C. (2009). Predicting missing links
       via local information. *European Physical Journal B*, 71(4),
       623-630.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import fraction_filter, resource_allocation_index
    >>> G = nx.les_miserables_graph()
    >>> H = resource_allocation_index(G)
    >>> backbone = fraction_filter(H, "resource_allocation", 0.3, ascending=False)
    >>> backbone.number_of_edges() <= H.number_of_edges()
    True
    """
    H = G.copy()
    nbrs = _neighbor_sets(G)
    deg = _degrees(nbrs)
    for u, v in H.edges():
        common = nbrs[u] & nbrs[v]
        score = sum(1.0 / deg[z] for z in common if deg[z] > 0)
        H[u][v]["resource_allocation"] = score
    return H


# =====================================================================
# Quasi-local methods
# =====================================================================


@not_implemented_for("multigraph")
def graph_distance_proximity(G, all_pairs=False):
    r"""Score node pairs by reciprocal shortest-path distance.

    The ``"dist"`` attribute holds the *proximity* :math:`1 / d(u, v)`,
    where :math:`d(u, v)` is the unweighted (hop-count) shortest-path
    distance from *u* to *v*, and 0 when *v* is unreachable from *u*.
    Despite its name it is not the distance itself.

    By default, scores are computed for existing edges only.  With
    ``all_pairs=True``, scores are computed for every node pair
    (all unordered pairs in undirected graphs; all ordered pairs in directed
    graphs, excluding self-pairs).

    Parameters
    ----------
    G : networkx.Graph or networkx.DiGraph
        Input graph.  Edge weights are ignored.
    all_pairs : bool, optional (default=False)
        If ``False``, annotate only existing edges in a copy of *G*.
        If ``True``, return a graph containing all node pairs as edges and
        annotate each with ``"dist"``.

    Returns
    -------
    H : graph
        Graph with the ``"dist"`` edge attribute (reciprocal distance)
        added.

    Raises
    ------
    NetworkXNotImplemented
        If *G* is a multigraph.

    Notes
    -----
    Graph distance is a link-prediction score for *unconnected* pairs [1]_.
    Every existing edge has distance 1, so with ``all_pairs=False`` every
    edge (other than a self-loop, which gets 0) receives ``"dist" == 1.0``
    and the score cannot discriminate between edges; filtering on it then
    keeps an arbitrary subset.  The method is only informative with
    ``all_pairs=True``.  Distances are not recomputed with the edge
    removed, because that is not the score defined in [1]_.

    References
    ----------
    .. [1] Liben-Nowell, D. & Kleinberg, J. (2007). The link-prediction
       problem for social networks. *JASIST*, 58(7), 1019-1031.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import graph_distance_proximity
    >>> G = nx.path_graph(4)
    >>> H = graph_distance_proximity(G)
    >>> {e: H.edges[e]["dist"] for e in H.edges()}
    {(0, 1): 1.0, (1, 2): 1.0, (2, 3): 1.0}
    >>> H_all = graph_distance_proximity(G, all_pairs=True)
    >>> H_all[0][3]["dist"]  # 1 / 3
    0.3333333333333333
    """
    sp = dict(nx.all_pairs_shortest_path_length(G))

    if not all_pairs:
        H = G.copy()
        for u, v in H.edges():
            d = sp.get(u, {}).get(v)
            H[u][v]["dist"] = 1.0 / d if d and d > 0 else 0.0
        return H

    H = G.__class__()
    H.add_nodes_from(G.nodes(data=True))
    nodes = list(G.nodes())

    if G.is_directed():
        pairs = ((u, v) for u in nodes for v in nodes if u != v)
    else:
        pairs = ((u, v) for i, u in enumerate(nodes) for v in nodes[i + 1 :])

    for u, v in pairs:
        d = sp.get(u, {}).get(v)
        dist_score = 1.0 / d if d and d > 0 else 0.0

        edge_data = {}
        if G.has_edge(u, v):
            edge_data.update(G[u][v])
        edge_data["dist"] = dist_score
        H.add_edge(u, v, **edge_data)

    return H


@not_implemented_for("multigraph")
def local_path_index(G, epsilon=0.01, weight=None):
    r"""Score each edge by the local path index.

    For each edge (u, v):

    .. math::

        \mathrm{LP}_{uv} = (A^2)_{uv} + \varepsilon \, (A^3)_{uv}

    where *A* is the adjacency matrix (without self-loops).  The score is
    stored as the ``"lp"`` edge attribute.

    Parameters
    ----------
    G : networkx.Graph or networkx.DiGraph
        Input graph.  For directed graphs, :math:`(A^k)_{uv}` counts the
        directed walks of length *k* from *u* to *v*.
    epsilon : float, optional (default=0.01)
        Weight for third-order paths.
    weight : string or None, optional (default=None)
        Edge attribute used as the entries of *A*.  ``None`` (the default)
        uses the unweighted 0/1 adjacency matrix, like the other proximity
        indices; pass e.g. ``"weight"`` for a weighted variant.

    Returns
    -------
    H : graph
        Copy of *G* with the ``"lp"`` edge attribute added.

    Raises
    ------
    NetworkXNotImplemented
        If *G* is a multigraph.

    References
    ----------
    .. [1] Lu, L., Jin, C.-H. & Zhou, T. (2009). Similarity index based on
       local paths for link prediction of complex networks. *Physical
       Review E*, 80(4), 046122.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import fraction_filter, local_path_index
    >>> G = nx.les_miserables_graph()
    >>> H = local_path_index(G)
    >>> backbone = fraction_filter(H, "lp", 0.3, ascending=False)
    >>> backbone.number_of_edges() <= H.number_of_edges()
    True
    """
    import numpy as np

    H = G.copy()
    if not H.number_of_edges():
        return H
    nodelist = list(G.nodes())
    G_noloops = G
    if nx.number_of_selfloops(G):
        G_noloops = G.copy()
        G_noloops.remove_edges_from(list(nx.selfloop_edges(G)))
    A = nx.to_scipy_sparse_array(
        G_noloops, nodelist=nodelist, weight=weight, format="csr"
    )
    A2 = A @ A
    S = (A2 + epsilon * (A2 @ A)).tocsr()
    node_idx = {node: i for i, node in enumerate(nodelist)}
    edges = list(H.edges())
    rows = np.fromiter((node_idx[u] for u, _ in edges), dtype=np.intp, count=len(edges))
    cols = np.fromiter((node_idx[v] for _, v in edges), dtype=np.intp, count=len(edges))
    values = np.asarray(S[rows, cols]).ravel()
    for (u, v), val in zip(edges, values):
        H[u][v]["lp"] = float(val)
    return H


_COMPLEXITY = {
    "neighborhood_overlap": {
        "time": "O(mn)",
        "space": "O(n + m)",
        "notes": "Worst-case over per-edge neighbor-set intersections.",
    },
    "jaccard_backbone": {
        "time": "O(mn)",
        "space": "O(n + m)",
    },
    "dice_backbone": {
        "time": "O(mn)",
        "space": "O(n + m)",
    },
    "cosine_backbone": {
        "time": "O(mn)",
        "space": "O(n + m)",
    },
    "hub_promoted_index": {
        "time": "O(mn)",
        "space": "O(n + m)",
    },
    "hub_depressed_index": {
        "time": "O(mn)",
        "space": "O(n + m)",
    },
    "lhn_local_index": {
        "time": "O(mn)",
        "space": "O(n + m)",
    },
    "preferential_attachment_score": {
        "time": "O(n + m)",
        "space": "O(n + m)",
    },
    "adamic_adar_index": {
        "time": "O(mn)",
        "space": "O(n + m)",
    },
    "resource_allocation_index": {
        "time": "O(mn)",
        "space": "O(n + m)",
    },
    "graph_distance_proximity": {
        "time": "O(n(n + m))",
        "space": "O(n^2 + m)",
        "notes": "Computes all-pairs shortest paths before edge annotation.",
    },
    "local_path_index": {
        "time": "O(n^3)",
        "space": "O(n^2)",
        "notes": "Worst case; sparse matrix products are much cheaper on sparse graphs.",
    },
}

append_complexity_docstrings(globals(), _COMPLEXITY)
