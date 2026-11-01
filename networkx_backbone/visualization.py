"""
Visualization utilities for backbone comparisons.

These helpers compare an original graph with a backbone graph and render
the differences using consistent defaults:

- removed nodes in red
- retained edges in thicker black
"""

import networkx as nx

from networkx_backbone._docstrings import append_complexity_docstrings

__all__ = [
    "graph_difference",
    "compare_graphs",
    "save_graph_comparison",
]


def _edge_key(u, v, directed):
    """Build a canonical edge key for directed or undirected matching.

    Undirected endpoints are ordered by their natural order when they are
    comparable and by ``(type name, repr)`` otherwise, so that e.g. ``1``
    and ``"1"`` are never confused and mixed-type labels never raise.
    """
    if directed:
        return (u, v)
    try:
        swap = v < u
    except TypeError:
        swap = (type(v).__name__, repr(v)) < (type(u).__name__, repr(u))
    return (v, u) if swap else (u, v)


def graph_difference(original, backbone):
    """Compute node/edge overlap and removals between two graphs.

    Parameters
    ----------
    original : graph
        Reference graph.
    backbone : graph
        Candidate backbone graph to compare against ``original``.

    Returns
    -------
    diff : dict
        Dictionary containing:

        - ``"kept_nodes"``: nodes of *original* also in *backbone*
        - ``"removed_nodes"``: nodes of *original* missing from *backbone*
        - ``"kept_edges"``: edges of *original* also in *backbone*
        - ``"removed_edges"``: edges of *original* missing from *backbone*

        Edges are tuples; for undirected graphs each edge appears once with
        its endpoints in a canonical order.

    Examples
    --------
    >>> import networkx as nx
    >>> from networkx_backbone import graph_difference
    >>> G = nx.path_graph(4)
    >>> backbone = nx.Graph([(1, 0)])
    >>> backbone.add_node(2)
    >>> diff = graph_difference(G, backbone)
    >>> diff["removed_nodes"], diff["kept_edges"]
    ({3}, {(0, 1)})
    >>> sorted(diff["removed_edges"])
    [(1, 2), (2, 3)]
    """
    directed = original.is_directed()

    original_nodes = set(original.nodes())
    backbone_nodes = set(backbone.nodes())
    kept_nodes = original_nodes & backbone_nodes
    removed_nodes = original_nodes - backbone_nodes

    original_edges = {_edge_key(u, v, directed) for u, v in original.edges()}
    backbone_edges = {_edge_key(u, v, directed) for u, v in backbone.edges()}
    kept_edges = original_edges & backbone_edges
    removed_edges = original_edges - kept_edges

    return {
        "kept_nodes": kept_nodes,
        "removed_nodes": removed_nodes,
        "kept_edges": kept_edges,
        "removed_edges": removed_edges,
    }


def compare_graphs(
    original,
    backbone,
    pos=None,
    ax=None,
    with_labels=False,
    node_size=220,
    kept_node_color="#f2f2f2",
    removed_node_color="red",
    base_edge_color="#c9c9c9",
    kept_edge_color="black",
    base_edge_width=0.8,
    kept_edge_width=2.6,
    title=None,
    return_diff=False,
    seed=None,
):
    """Visualize differences between an original graph and a backbone graph.

    Original edges are drawn as a thin grey layer, edges retained in the
    backbone are drawn on top in thicker black, and nodes missing from the
    backbone are drawn in red.

    Parameters
    ----------
    original : graph
        Reference graph.
    backbone : graph
        Backbone graph to compare against ``original``.
    pos : dict or None, optional (default=None)
        Node-position dictionary. If ``None``, uses
        ``nx.spring_layout(original, seed=seed)``.  Pass the same *pos*
        to several calls to draw comparable panels.
    ax : matplotlib Axes or None, optional (default=None)
        Axis to draw on. If ``None``, a new figure and axis are created
        with ``plt.subplots``; the caller owns (and should close) it.
    with_labels : bool, optional (default=False)
        Whether to draw node labels.
    node_size : int, optional (default=220)
        Node size.
    kept_node_color : color, optional
        Color for nodes retained in the backbone.
    removed_node_color : color, optional (default="red")
        Color for nodes removed from the backbone.
    base_edge_color : color, optional
        Base color for original edges.
    kept_edge_color : color, optional (default="black")
        Color for retained backbone edges.
    base_edge_width : float, optional
        Width for base/original edge layer.
    kept_edge_width : float, optional (default=2.6)
        Width for retained backbone edges (drawn on top).
    title : str or None, optional
        Plot title.
    return_diff : bool, optional (default=False)
        If ``True``, return the computed graph-difference dictionary.
    seed : integer, random_state, or None (default)
        Seed for the spring layout used when *pos* is ``None``.  Pass an
        integer for a reproducible layout.

    Returns
    -------
    (fig, ax) or (fig, ax, diff)
        Matplotlib figure/axes, and optionally the difference dictionary
        returned by :func:`graph_difference`.

    Notes
    -----
    The function never calls ``plt.show()``.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> import matplotlib.pyplot as plt
    >>> import networkx as nx
    >>> from networkx_backbone import boolean_filter, compare_graphs, metric_backbone
    >>> G = nx.les_miserables_graph()
    >>> backbone = boolean_filter(metric_backbone(G), "metric_keep")
    >>> fig, ax = plt.subplots()
    >>> fig2, ax2 = compare_graphs(G, backbone, ax=ax, seed=1, title="Metric backbone")
    >>> fig2 is fig, ax.get_title()
    (True, 'Metric backbone')
    >>> plt.close(fig)
    """
    import matplotlib.pyplot as plt

    if pos is None:
        pos = nx.spring_layout(original, seed=seed)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    diff = graph_difference(original, backbone)
    directed = original.is_directed()

    original_edge_tuples = list(original.edges())
    kept_edge_tuples = [
        (u, v)
        for u, v in original_edge_tuples
        if _edge_key(u, v, directed) in diff["kept_edges"]
    ]

    kept_nodes = [n for n in original.nodes() if n in diff["kept_nodes"]]
    removed_nodes = [n for n in original.nodes() if n in diff["removed_nodes"]]

    nx.draw_networkx_edges(
        original,
        pos,
        ax=ax,
        edgelist=original_edge_tuples,
        edge_color=base_edge_color,
        width=base_edge_width,
        alpha=0.9,
        arrows=directed,
    )
    nx.draw_networkx_edges(
        original,
        pos,
        ax=ax,
        edgelist=kept_edge_tuples,
        edge_color=kept_edge_color,
        width=kept_edge_width,
        alpha=1.0,
        arrows=directed,
    )
    nx.draw_networkx_nodes(
        original,
        pos,
        ax=ax,
        nodelist=kept_nodes,
        node_color=kept_node_color,
        node_size=node_size,
        linewidths=0.8,
        edgecolors="black",
    )
    if removed_nodes:
        nx.draw_networkx_nodes(
            original,
            pos,
            ax=ax,
            nodelist=removed_nodes,
            node_color=removed_node_color,
            node_size=node_size,
            linewidths=0.8,
            edgecolors="black",
        )

    if with_labels:
        nx.draw_networkx_labels(original, pos, ax=ax, font_size=7)

    if title:
        ax.set_title(title)
    ax.set_axis_off()

    if return_diff:
        return fig, ax, diff
    return fig, ax


def save_graph_comparison(
    original,
    backbone,
    output_path,
    pos=None,
    dpi=180,
    bbox_inches="tight",
    **kwargs,
):
    """Save a comparison visualization to disk.

    Draws the comparison with :func:`compare_graphs` and writes it with
    ``Figure.savefig``.

    Parameters
    ----------
    original : graph
        Reference graph.
    backbone : graph
        Backbone graph to compare against ``original``.
    output_path : str or path-like
        Destination file; the format is inferred from the extension
        (e.g. ``.png``, ``.pdf``, ``.svg``).
    pos : dict or None, optional (default=None)
        Node positions, passed to :func:`compare_graphs`.
    dpi : float, optional (default=180)
        Resolution passed to ``savefig``.
    bbox_inches : str or None, optional (default="tight")
        Bounding-box option passed to ``savefig``.
    **kwargs
        Further keyword arguments for :func:`compare_graphs` (e.g.
        ``ax``, ``title``, ``with_labels``, ``seed``, ``return_diff``).

    Returns
    -------
    output_path or (output_path, diff)
        The path that was written, plus the :func:`graph_difference`
        dictionary when ``return_diff=True``.

    Notes
    -----
    If no ``ax`` is given, the figure created internally is closed after
    saving, so repeated calls do not accumulate open figures.  A figure
    owned by the caller (passed via ``ax``) is left open.

    Examples
    --------
    >>> import os, tempfile
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> import networkx as nx
    >>> from networkx_backbone import save_graph_comparison
    >>> G = nx.cycle_graph(5)
    >>> backbone = nx.path_graph(5)
    >>> path = os.path.join(tempfile.mkdtemp(), "comparison.png")
    >>> out, diff = save_graph_comparison(G, backbone, path, seed=1, return_diff=True)
    >>> os.path.exists(out), sorted(diff["removed_edges"])
    (True, [(0, 4)])
    """
    import matplotlib.pyplot as plt

    owns_figure = kwargs.get("ax") is None
    result = compare_graphs(original, backbone, pos=pos, **kwargs)
    fig = result[0]
    try:
        fig.savefig(output_path, dpi=dpi, bbox_inches=bbox_inches)
    finally:
        if owns_figure:
            plt.close(fig)
    if kwargs.get("return_diff", False):
        return output_path, result[2]
    return output_path


_COMPLEXITY = {
    "graph_difference": {
        "time": "O(n + m)",
        "space": "O(n + m)",
    },
    "compare_graphs": {
        "time": "O(n + m) without layout; plus layout cost if pos is None",
        "space": "O(n + m)",
        "notes": "Spring layout adds iterative graph-drawing overhead.",
    },
    "save_graph_comparison": {
        "time": "O(n + m) without layout; plus rendering/write cost",
        "space": "O(n + m)",
    },
}

append_complexity_docstrings(globals(), _COMPLEXITY)
