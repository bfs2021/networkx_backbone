"""
Hybrid backbone extraction methods.

These methods combine statistical and structural approaches.

Methods
-------
glab_filter
    Zhang et al. (2014) -- Globally and Locally Adaptive Backbone.
    Tests each link's shortest-path involvement against a local,
    degree-dependent null model.
"""

import math

import networkx as nx
from networkx.utils import not_implemented_for

from networkx_backbone._docstrings import append_complexity_docstrings

__all__ = ["glab_filter"]


@not_implemented_for("directed")
@not_implemented_for("multigraph")
def glab_filter(G, weight="weight", c=0.5):
    r"""Compute GLAB (Globally and Locally Adaptive Backbone) p-values.

    The GLAB method [1]_ (called GLANB in the paper) combines global and local
    information.  For every node *i* and incident link *(i, j)* it computes the
    *involvement* of the link: the fraction of shortest paths from *i* to every
    other node that leave *i* through *(i, j)*,

    .. math::

        I_{ij} = \frac{1}{N - 1} \sum_{s \neq i}
                 \frac{g_{is}(i, j)}{g_{is}},

    where :math:`g_{is}` is the number of shortest paths between *i* and *s*
    and :math:`g_{is}(i, j)` the number of those that use link *(i, j)*.  Link
    lengths are ``1 / weight``, so heavy links are short.  Involvements sum to
    one over each node's links.  Under a null model in which a node's
    involvement is spread uniformly over its :math:`k_i` links, the
    statistical importance is

    .. math::

        SI_{ij} = (1 - I_{ij})^{(k_i - 1)^c}, \qquad k_i \geq 2,

    and the p-value of an undirected link is :math:`\min(SI_{ij}, SI_{ji})`.
    When one endpoint has degree 1, only the other endpoint's value is used.

    Parameters
    ----------
    G : networkx.Graph
        An undirected NetworkX graph.
    weight : string or None, optional (default="weight")
        Edge attribute holding positive weights; shortest paths use
        ``1 / weight`` as link length.  If ``None``, every link has length 1.
    c : float, optional (default=0.5)
        Non-negative exponent controlling how strongly the degree of a node
        enters the null model.  ``c=0`` ignores degree.

    Returns
    -------
    H : networkx.Graph
        A copy of *G* with a ``"glab_pvalue"`` edge attribute.

    Raises
    ------
    NetworkXNotImplemented
        If *G* is directed or a multigraph.
    NetworkXError
        If ``weight`` is given and an edge has a missing, non-positive, or
        non-finite weight.
    ValueError
        If ``c`` is negative.

    See Also
    --------
    disparity_filter : Purely local significance test.
    high_salience_skeleton : Purely global shortest-path-tree method.

    Notes
    -----
    In a disconnected graph the involvement of node *i* is normalized by the
    number of nodes reachable from *i* rather than :math:`N - 1`, so the
    involvements of each node still sum to one.  A link whose endpoints both
    have degree 1 (an isolated dyad) has no null distribution and gets a
    p-value of 1.

    The paper also defines a directed variant using out- and in-involvement;
    it is not implemented yet.

    References
    ----------
    .. [1] Zhang, X., Zhang, Z., Zhao, H., Wang, Q., & Zhu, J. (2014).
       Extracting the globally and locally adaptive backbone of complex
       networks. *PLoS ONE*, 9(6), e100428.
       https://doi.org/10.1371/journal.pone.0100428

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import glab_filter, threshold_filter
    >>> G = nx.les_miserables_graph()
    >>> H = glab_filter(G)
    >>> backbone = threshold_filter(H, "glab_pvalue", 0.05, mode="below")
    >>> backbone.number_of_edges() <= H.number_of_edges()
    True
    """
    if c < 0:
        raise ValueError(f"c must be non-negative, got {c}")

    H = G.copy()
    if weight is not None:
        for u, v, data in G.edges(data=True):
            w = data.get(weight)
            if w is None or not (0 < w < math.inf):
                raise nx.NetworkXError(
                    f"Edge ({u}, {v}) has a missing, non-positive, or non-finite "
                    f"{weight!r} attribute."
                )

        def length(u, v, data):
            return 1.0 / data[weight]

    else:
        length = None

    degree = dict(G.degree())
    # si[(i, j)] holds SI_ij: the importance of link (i, j) seen from node i.
    si = {}
    for i in G:
        if degree[i] < 2:
            continue
        involvement = _involvement(G, i, length)
        exponent = (degree[i] - 1) ** c
        for j, inv in involvement.items():
            si[(i, j)] = max(0.0, 1.0 - inv) ** exponent

    for u, v, data in H.edges(data=True):
        values = [si[key] for key in ((u, v), (v, u)) if key in si]
        data["glab_pvalue"] = min(values) if values else 1.0

    return H


def _involvement(G, source, length):
    """Fraction of shortest paths from ``source`` that leave via each neighbor.

    Returns ``{neighbor: I_source,neighbor}``, normalized by the number of
    nodes reachable from ``source``.
    """
    if length is None:
        pred = nx.predecessor(G, source)
        order = list(pred)  # BFS discovery order, nondecreasing distance
    else:
        pred, dist = nx.dijkstra_predecessor_and_distance(G, source, weight=length)
        order = sorted(dist, key=dist.__getitem__)

    sigma = {source: 1}
    # first_hop[v][j]: number of shortest source->v paths whose first link is
    # (source, j).
    first_hop = {}
    for v in order:
        if v == source:
            continue
        sigma[v] = sum(sigma[p] for p in pred[v])
        counts = {}
        for p in pred[v]:
            if p == source:
                counts[v] = counts.get(v, 0) + 1
            else:
                for j, cnt in first_hop[p].items():
                    counts[j] = counts.get(j, 0) + cnt
        first_hop[v] = counts

    reachable = len(order) - 1
    involvement = dict.fromkeys(G[source], 0.0)
    if reachable == 0:
        return involvement
    for v, counts in first_hop.items():
        total = sigma[v]
        for j, cnt in counts.items():
            involvement[j] += cnt / total
    return {j: val / reachable for j, val in involvement.items()}


_COMPLEXITY = {
    "glab_filter": {
        "time": "O(n m + n^2 log n)",
        "space": "O(n k_max + m)",
        "notes": "One single-source shortest-path search per node.",
    }
}

append_complexity_docstrings(globals(), _COMPLEXITY)
