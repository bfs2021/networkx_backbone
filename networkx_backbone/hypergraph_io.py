"""
Hypergraph ingestion and interoperability adapters.

``networkx_backbone`` represents a hypergraph as a plain iterable of hyperedges
(each an iterable of node labels), which every hypergraph backbone method in
:mod:`networkx_backbone.hypergraph` consumes.  This module converts that
representation to and from:

- a **NetworkX incidence bipartite graph**, so bipartite projection backbones
  (:func:`~networkx_backbone.sdsm`, :func:`~networkx_backbone.fdsm`, ...) can be
  applied to hypergraphs;
- the **HIF** (Hypergraph Interchange Format) JSON standard;
- the **xgi**, **HyperNetX**, **HypergraphX**, and **Hypergraph Analysis
  Toolbox (HAT)** hypergraph classes.

The third-party libraries are imported lazily; none is a required dependency.
HIF is the recommended interchange path because all four libraries read and
write it, so it needs only the standard library.
"""

import importlib
import json
import os

import networkx as nx

from networkx_backbone.hypergraph import _sorted_labels

__all__ = [
    "hypergraph_to_bipartite",
    "read_hif",
    "write_hif",
    "from_xgi",
    "to_xgi",
    "from_hypernetx",
    "to_hypernetx",
    "from_hypergraphx",
    "to_hypergraphx",
    "from_hat",
    "to_hat",
]

#: HIF ``network-type`` values describing an undirected hypergraph.
_UNDIRECTED_HIF_TYPES = ("undirected", "asc")


def _as_hyperedges(hyperedges):
    """Normalise to a list of non-empty :class:`frozenset` hyperedges (order kept)."""
    out = []
    for e in hyperedges:
        fs = frozenset(e)
        if fs:
            out.append(fs)
    return out


def _require(module, pip_name=None):
    """Import an optional dependency or raise a helpful ImportError."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(
            f"'{module}' is required for this conversion; install it with "
            f"`pip install {pip_name or module}`"
        ) from exc


# ---------------------------------------------------------------------------
# NetworkX incidence bipartite graph
# ---------------------------------------------------------------------------


def hypergraph_to_bipartite(hyperedges, edge_prefix="he"):
    """Convert a hypergraph to its incidence bipartite graph.

    Nodes of the hypergraph form one partition (``bipartite=0``); hyperedges form
    the other (``bipartite=1``).  The ``i``-th (non-empty) hyperedge becomes the
    node ``(edge_prefix, i)`` -- a tuple label, so it cannot be confused with a
    string node label such as ``"he0"`` -- carrying a ``"members"`` attribute
    with the hyperedge as a :class:`frozenset`.  The result is exactly the input
    expected by the bipartite projection backbones, so a hypergraph can be
    backboned with, e.g., :func:`~networkx_backbone.sdsm` or
    :func:`~networkx_backbone.fdsm`.

    Parameters
    ----------
    hyperedges : iterable of iterables
        The hypergraph.  Empty hyperedges are skipped.
    edge_prefix : hashable, optional (default="he")
        First element of the ``(edge_prefix, i)`` hyperedge-node labels.

    Returns
    -------
    B : networkx.Graph
        The incidence bipartite graph.
    nodes : list
        The hypergraph's nodes (the agent partition), suitable to pass as
        ``agent_nodes`` to the bipartite methods.  Sorted in natural order when
        the labels are mutually comparable, otherwise by ``repr``.

    Raises
    ------
    ValueError
        If a hyperedge label ``(edge_prefix, i)`` coincides with a hypergraph
        node label (choose a different *edge_prefix*).

    See Also
    --------
    sdsm, fdsm : Bipartite projection backbones that consume the result.
    intersection_graph : Hyperedge-level (line-graph) view of a hypergraph.

    Examples
    --------
    >>> import networkx as nx
    >>> import networkx_backbone as nb
    >>> B, nodes = nb.hypergraph_to_bipartite([(1, 2, 3), (2, 3, 4), (5, 6)])
    >>> nx.is_bipartite(B)
    True
    >>> nodes
    [1, 2, 3, 4, 5, 6]
    >>> sorted(B[("he", 0)])
    [1, 2, 3]
    >>> scored = nb.sdsm(B, agent_nodes=nodes)
    """
    edges = _as_hyperedges(hyperedges)
    nodes = _sorted_labels({v for e in edges for v in e})
    node_set = set(nodes)

    edge_labels = [(edge_prefix, i) for i in range(len(edges))]
    clashes = [label for label in edge_labels if label in node_set]
    if clashes:
        raise ValueError(
            f"hyperedge label(s) {clashes[:3]!r} collide with hypergraph node "
            "labels; pass a different edge_prefix"
        )

    B = nx.Graph()
    B.add_nodes_from(nodes, bipartite=0)
    for label, e in zip(edge_labels, edges):
        B.add_node(label, bipartite=1, members=e)
        for v in e:
            B.add_edge(v, label)
    return B, nodes


# ---------------------------------------------------------------------------
# HIF (Hypergraph Interchange Format)
# ---------------------------------------------------------------------------


def _to_builtin(value):
    """Convert numpy scalars (also inside tuples) to builtin Python values."""
    if isinstance(value, tuple):
        return tuple(_to_builtin(v) for v in value)
    if type(value).__module__ == "numpy" and getattr(value, "ndim", None) == 0:
        return value.item()
    return value


def _from_json(value):
    """Make a JSON-decoded identifier hashable (JSON arrays -> tuples)."""
    if isinstance(value, list):
        return tuple(_from_json(v) for v in value)
    return value


def _check_network_type(network_type):
    if network_type in _UNDIRECTED_HIF_TYPES:
        return
    if network_type == "directed":
        raise nx.NetworkXNotImplemented(
            "directed hypergraphs are not supported; networkx_backbone "
            "hypergraphs are undirected (HIF network-type 'undirected' or 'asc')"
        )
    raise ValueError(
        "HIF network-type must be 'undirected', 'asc', or 'directed', "
        f"got {network_type!r}"
    )


def write_hif(hyperedges, target=None, weights=None, network_type="undirected"):
    """Serialise a hypergraph to the HIF (Hypergraph Interchange Format) standard.

    HIF is a JSON format shared by xgi, HyperNetX, HypergraphX, and HAT, making it
    the most portable interchange path.  Numpy scalar node labels and weights are
    converted to builtin Python numbers so the result is JSON-serialisable.
    Tuple node labels are written as JSON arrays and restored as tuples by
    :func:`read_hif`.

    Parameters
    ----------
    hyperedges : iterable of iterables
        The hypergraph.  Empty hyperedges are skipped.
    target : str, path, file-like, or None, optional (default=None)
        Where to write.  If ``None``, the HIF dictionary is returned instead of
        being written.
    weights : iterable of numbers or None, optional (default=None)
        Optional per-hyperedge weights, written to the HIF ``edges`` list.
    network_type : {"undirected", "asc"}, optional (default="undirected")
        Value for the HIF ``network-type`` field.

    Returns
    -------
    result : dict or target
        The HIF dictionary if *target* is ``None``; otherwise *target*.

    Raises
    ------
    ValueError
        If *weights* does not match *hyperedges* in length, or *network_type*
        is not a valid HIF network type.
    networkx.NetworkXNotImplemented
        If *network_type* is ``"directed"`` (hyperedges here are undirected).

    See Also
    --------
    read_hif : The inverse operation.

    Examples
    --------
    >>> import networkx_backbone as nb
    >>> hif = nb.write_hif([(1, 2, 3), (3, 4)], weights=[2, 1])
    >>> hif["network-type"]
    'undirected'
    >>> len(hif["incidences"])
    5
    >>> hif["edges"]
    [{'edge': 0, 'weight': 2}, {'edge': 1, 'weight': 1}]
    """
    _check_network_type(network_type)
    raw = list(hyperedges)
    if weights is not None:
        weights = list(weights)
        if len(weights) != len(raw):
            raise ValueError("weights must have the same length as hyperedges")

    incidences = []
    edge_records = []
    edge_id = 0
    for i, e in enumerate(raw):
        members = frozenset(_to_builtin(v) for v in e)
        if not members:
            continue
        for node in _sorted_labels(members):
            incidences.append({"edge": edge_id, "node": node})
        if weights is not None:
            edge_records.append({"edge": edge_id, "weight": _to_builtin(weights[i])})
        edge_id += 1

    hif = {"network-type": network_type, "metadata": {}, "incidences": incidences}
    if edge_records:
        hif["edges"] = edge_records

    if target is None:
        return hif
    if hasattr(target, "write"):
        json.dump(hif, target)
        return target
    with open(target, "w") as handle:
        json.dump(hif, handle)
    return target


def read_hif(source, return_weights=False):
    """Read a hypergraph from the HIF (Hypergraph Interchange Format) standard.

    JSON arrays used as node or edge identifiers are converted to tuples so they
    are hashable (this restores tuple node labels written by :func:`write_hif`).

    Parameters
    ----------
    source : dict, str, path, or file-like
        A parsed HIF dictionary, a path to a HIF JSON file, or an open file.
    return_weights : bool, optional (default=False)
        If ``True``, also return per-hyperedge weights, read from each ``edges``
        record's ``weight`` (or ``attrs["weight"]``) and defaulting to 1.

    Returns
    -------
    hyperedges : list of frozenset
        The hyperedges, in order of first appearance among the incidences.
    weights : list of numbers
        Returned only when *return_weights* is ``True``.

    Raises
    ------
    TypeError
        If *source* is not a dict, path, or file-like object.
    networkx.NetworkXNotImplemented
        If the HIF ``network-type`` is ``"directed"`` (directed hyperedges would
        otherwise silently lose their head/tail orientation).
    ValueError
        If the HIF ``network-type`` is not a recognised value.

    See Also
    --------
    write_hif : The inverse operation.

    Examples
    --------
    >>> import networkx_backbone as nb
    >>> hif = nb.write_hif([(1, 2, 3), (3, 4)], weights=[2, 1])
    >>> edges, weights = nb.read_hif(hif, return_weights=True)
    >>> [sorted(e) for e in edges], weights
    ([[1, 2, 3], [3, 4]], [2, 1])
    """
    if isinstance(source, dict):
        hif = source
    elif isinstance(source, (str, os.PathLike)):
        with open(source) as handle:
            hif = json.load(handle)
    elif hasattr(source, "read"):
        hif = json.load(source)
    else:
        raise TypeError("source must be a dict, path, or file-like object")

    _check_network_type(hif.get("network-type", "undirected"))

    groups = {}
    for incidence in hif.get("incidences", []):
        edge = _from_json(incidence["edge"])
        groups.setdefault(edge, set()).add(_from_json(incidence["node"]))

    edge_ids = list(groups)
    hyperedges = [frozenset(groups[eid]) for eid in edge_ids]
    if not return_weights:
        return hyperedges

    weight_map = {}
    for rec in hif.get("edges", []):
        weight = rec.get("weight")
        if weight is None:
            weight = (rec.get("attrs") or {}).get("weight", 1)
        weight_map[_from_json(rec["edge"])] = weight
    weights = [weight_map.get(eid, 1) for eid in edge_ids]
    return hyperedges, weights


# ---------------------------------------------------------------------------
# xgi
# ---------------------------------------------------------------------------


def from_xgi(H):
    """Convert an :class:`xgi.Hypergraph` to a list of :class:`frozenset` hyperedges.

    Parameters
    ----------
    H : xgi.Hypergraph
        The xgi hypergraph (read via ``H.edges.members()``).

    Returns
    -------
    hyperedges : list of frozenset
        One frozenset per xgi hyperedge, in xgi's edge order.

    See Also
    --------
    to_xgi : The inverse conversion.
    read_hif : Library-independent interchange via HIF.

    Examples
    --------
    >>> import xgi  # doctest: +SKIP
    >>> import networkx_backbone as nb
    >>> nb.from_xgi(xgi.Hypergraph([[1, 2, 3], [3, 4]]))  # doctest: +SKIP
    [frozenset({1, 2, 3}), frozenset({3, 4})]
    """
    return [frozenset(members) for members in H.edges.members()]


def to_xgi(hyperedges):
    """Convert hyperedges to an :class:`xgi.Hypergraph`.

    Parameters
    ----------
    hyperedges : iterable of iterables
        The hypergraph.  Empty hyperedges are skipped.

    Returns
    -------
    H : xgi.Hypergraph
        The equivalent xgi hypergraph.

    Raises
    ------
    ImportError
        If ``xgi`` is not installed.

    See Also
    --------
    from_xgi : The inverse conversion.
    write_hif : Library-independent interchange via HIF.

    Examples
    --------
    >>> import networkx_backbone as nb
    >>> H = nb.to_xgi([(1, 2, 3), (3, 4)])  # doctest: +SKIP
    >>> H.num_edges  # doctest: +SKIP
    2
    """
    xgi = _require("xgi")
    return xgi.Hypergraph([set(e) for e in _as_hyperedges(hyperedges)])


# ---------------------------------------------------------------------------
# HyperNetX
# ---------------------------------------------------------------------------


def from_hypernetx(H):
    """Convert a :class:`hypernetx.Hypergraph` to a list of :class:`frozenset` hyperedges.

    Parameters
    ----------
    H : hypernetx.Hypergraph
        The HyperNetX hypergraph (read via ``H.incidence_dict``).

    Returns
    -------
    hyperedges : list of frozenset
        One frozenset per HyperNetX edge, in ``incidence_dict`` order (edge
        names are discarded).

    See Also
    --------
    to_hypernetx : The inverse conversion.
    read_hif : Library-independent interchange via HIF.

    Examples
    --------
    >>> import hypernetx as hnx  # doctest: +SKIP
    >>> import networkx_backbone as nb
    >>> nb.from_hypernetx(hnx.Hypergraph({"a": [1, 2], "b": [2, 3]}))  # doctest: +SKIP
    [frozenset({1, 2}), frozenset({2, 3})]
    """
    return [frozenset(members) for members in H.incidence_dict.values()]


def to_hypernetx(hyperedges):
    """Convert hyperedges to a :class:`hypernetx.Hypergraph`.

    Parameters
    ----------
    hyperedges : iterable of iterables
        The hypergraph.  Empty hyperedges are skipped; the remaining edges are
        named ``0, 1, 2, ...``.

    Returns
    -------
    H : hypernetx.Hypergraph
        The equivalent HyperNetX hypergraph.

    Raises
    ------
    ImportError
        If ``hypernetx`` is not installed.

    See Also
    --------
    from_hypernetx : The inverse conversion.
    write_hif : Library-independent interchange via HIF.

    Examples
    --------
    >>> import networkx_backbone as nb
    >>> H = nb.to_hypernetx([(1, 2, 3), (3, 4)])  # doctest: +SKIP
    >>> sorted(H.incidence_dict)  # doctest: +SKIP
    [0, 1]
    """
    hnx = _require("hypernetx")
    return hnx.Hypergraph({i: set(e) for i, e in enumerate(_as_hyperedges(hyperedges))})


# ---------------------------------------------------------------------------
# HypergraphX
# ---------------------------------------------------------------------------


def from_hypergraphx(H):
    """Convert a ``hypergraphx.Hypergraph`` to a list of :class:`frozenset` hyperedges.

    Parameters
    ----------
    H : hypergraphx.Hypergraph
        The HypergraphX hypergraph (read via ``H.get_edges()``).

    Returns
    -------
    hyperedges : list of frozenset
        One frozenset per HypergraphX hyperedge.

    See Also
    --------
    to_hypergraphx : The inverse conversion.
    read_hif : Library-independent interchange via HIF.

    Examples
    --------
    >>> import hypergraphx as hgx  # doctest: +SKIP
    >>> import networkx_backbone as nb
    >>> nb.from_hypergraphx(hgx.Hypergraph(edge_list=[(1, 2), (2, 3, 4)]))  # doctest: +SKIP
    [frozenset({1, 2}), frozenset({2, 3, 4})]
    """
    return [frozenset(e) for e in H.get_edges()]


def to_hypergraphx(hyperedges):
    """Convert hyperedges to a ``hypergraphx.Hypergraph``.

    Parameters
    ----------
    hyperedges : iterable of iterables
        The hypergraph.  Empty hyperedges are skipped.

    Returns
    -------
    H : hypergraphx.Hypergraph
        The equivalent HypergraphX hypergraph (built from an ``edge_list`` of
        tuples).

    Raises
    ------
    ImportError
        If ``hypergraphx`` is not installed.

    See Also
    --------
    from_hypergraphx : The inverse conversion.
    write_hif : Library-independent interchange via HIF.

    Examples
    --------
    >>> import networkx_backbone as nb
    >>> H = nb.to_hypergraphx([(1, 2, 3), (3, 4)])  # doctest: +SKIP
    >>> H.num_edges()  # doctest: +SKIP
    2
    """
    hgx = _require("hypergraphx")
    return hgx.Hypergraph(edge_list=[tuple(e) for e in _as_hyperedges(hyperedges)])


# ---------------------------------------------------------------------------
# Hypergraph Analysis Toolbox (HAT)
# ---------------------------------------------------------------------------


def _hat_incidence_matrix(H):
    for attr in ("IM", "incidence_matrix", "incidenceMatrix"):
        value = getattr(H, attr, None)
        if value is None:
            continue
        value = value() if callable(value) else value
        if hasattr(value, "toarray"):  # scipy.sparse
            value = value.toarray()
        return value
    raise AttributeError("could not find an incidence matrix on the HAT hypergraph")


def from_hat(H):
    """Convert a HAT ``Hypergraph`` to a list of :class:`frozenset` hyperedges.

    Node labels are positional (row indices of the incidence matrix), as HAT is
    incidence/tensor-based.

    Parameters
    ----------
    H : HAT.Hypergraph
        The HAT hypergraph.  Its node-by-edge incidence matrix is read from the
        ``IM``, ``incidence_matrix``, or ``incidenceMatrix`` attribute (dense or
        scipy sparse; a method is called).

    Returns
    -------
    hyperedges : list of frozenset of int
        One frozenset of row indices per non-empty incidence-matrix column.

    Raises
    ------
    ImportError
        If ``numpy`` is not installed.
    AttributeError
        If no incidence matrix can be found on *H*.

    See Also
    --------
    to_hat : The inverse conversion.
    read_hif : Library-independent interchange via HIF.

    Examples
    --------
    >>> import networkx_backbone as nb
    >>> H = nb.to_hat([("a", "b"), ("b", "c", "d")])  # doctest: +SKIP
    >>> nb.from_hat(H)  # doctest: +SKIP
    [frozenset({0, 1}), frozenset({1, 2, 3})]
    """
    np = _require("numpy")
    matrix = np.asarray(_hat_incidence_matrix(H))
    hyperedges = []
    for col in range(matrix.shape[1]):
        members = frozenset(int(r) for r in np.flatnonzero(matrix[:, col]))
        if members:
            hyperedges.append(members)
    return hyperedges


def to_hat(hyperedges):
    """Convert hyperedges to a HAT ``Hypergraph`` via an incidence matrix.

    Node labels are replaced by contiguous integer indices (HAT is positional),
    assigned in sorted label order (natural order when the labels are mutually
    comparable, otherwise by ``repr``).

    Parameters
    ----------
    hyperedges : iterable of iterables
        The hypergraph.  Empty hyperedges are skipped.

    Returns
    -------
    H : HAT.Hypergraph
        A HAT hypergraph built from the ``nodes x hyperedges`` 0/1 incidence
        matrix (``HAT.Hypergraph(incidence_matrix=...)``).

    Raises
    ------
    ImportError
        If ``numpy`` or ``HAT`` (``pip install HypergraphAnalysisToolbox``) is
        not installed.

    See Also
    --------
    from_hat : The inverse conversion.
    write_hif : Library-independent interchange via HIF.

    Examples
    --------
    >>> import networkx_backbone as nb
    >>> H = nb.to_hat([("a", "b"), ("b", "c", "d")])  # doctest: +SKIP
    """
    np = _require("numpy")
    hat = _require("HAT", pip_name="HypergraphAnalysisToolbox")

    edges = _as_hyperedges(hyperedges)
    nodes = _sorted_labels({v for e in edges for v in e})
    index = {v: i for i, v in enumerate(nodes)}
    matrix = np.zeros((len(nodes), len(edges)), dtype=int)
    for j, e in enumerate(edges):
        for v in e:
            matrix[index[v], j] = 1
    return hat.Hypergraph(incidence_matrix=matrix)
