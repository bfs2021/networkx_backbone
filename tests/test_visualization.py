"""Tests for visualization utilities."""

import networkx as nx
import pytest

from networkx_backbone import compare_graphs, graph_difference, save_graph_comparison


def test_graph_difference_nodes_and_edges():
    G = nx.path_graph([0, 1, 2, 3])
    H = nx.Graph()
    H.add_nodes_from([0, 1, 2])
    H.add_edge(0, 1)

    diff = graph_difference(G, H)

    assert diff["removed_nodes"] == {3}
    assert diff["kept_nodes"] == {0, 1, 2}
    assert diff["kept_edges"] == {(0, 1)}
    assert diff["removed_edges"] == {(1, 2), (2, 3)}


def test_graph_difference_directed_orientation():
    G = nx.DiGraph()
    G.add_edges_from([(0, 1), (1, 0), (1, 2)])
    H = nx.DiGraph()
    H.add_edges_from([(0, 1)])

    diff = graph_difference(G, H)
    assert diff["kept_edges"] == {(0, 1)}
    assert diff["removed_edges"] == {(1, 0), (1, 2)}


def test_compare_graphs_returns_diff():
    pytest.importorskip("matplotlib")

    G = nx.cycle_graph(4)
    H = nx.path_graph([0, 1, 2])

    fig, ax, diff = compare_graphs(G, H, return_diff=True)
    assert fig is not None
    assert ax is not None
    assert 3 in diff["removed_nodes"]


# ---------------------------------------------------------------------
# Edge keys
# ---------------------------------------------------------------------


def test_graph_difference_does_not_confuse_int_and_str_labels():
    G = nx.Graph([(1, "1"), (1, 2), ("1", 2)])
    H = nx.Graph([("1", 1), ("1", 2)])
    diff = graph_difference(G, H)
    assert len(diff["kept_edges"]) == 2
    assert len(diff["removed_edges"]) == 1
    (removed,) = diff["removed_edges"]
    assert set(removed) == {1, 2}
    assert all(type(x) is int for x in removed)


def test_graph_difference_mixed_types_and_order_independence():
    G = nx.Graph([("a", 1), (2, 10), ((0, 1), 2.5)])
    H = nx.Graph([(1, "a"), (10, 2)])
    diff = graph_difference(G, H)
    assert len(diff["kept_edges"]) == 2
    assert (2, 10) in diff["kept_edges"]  # natural order when comparable
    assert len(diff["removed_edges"]) == 1


# ---------------------------------------------------------------------
# Plotting (Agg backend)
# ---------------------------------------------------------------------


@pytest.fixture
def plt(monkeypatch):
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    import matplotlib.pyplot as plt

    def _no_show(*args, **kwargs):  # pragma: no cover - must never be called
        raise AssertionError("plt.show() must not be called")

    monkeypatch.setattr(plt, "show", _no_show)
    yield plt
    plt.close("all")


def test_compare_graphs_uses_given_ax(plt):
    G = nx.cycle_graph(5)
    H = nx.path_graph(4)
    fig, ax = plt.subplots()
    n_open = len(plt.get_fignums())
    out_fig, out_ax = compare_graphs(G, H, ax=ax, title="Backbone", with_labels=True)
    assert out_fig is fig and out_ax is ax
    assert ax.get_title() == "Backbone"
    assert {t.get_text() for t in ax.texts} == {str(n) for n in G}
    assert len(plt.get_fignums()) == n_open


def test_compare_graphs_new_figure_pos_and_seed(plt):
    G = nx.karate_club_graph()
    H = nx.Graph(list(G.edges())[:10])
    pos = nx.circular_layout(G)
    fig, ax = compare_graphs(G, H, pos=pos)
    assert ax.figure is fig
    assert ax.get_title() == ""
    # Removed nodes are drawn in a second node collection.
    assert len(ax.collections) == 4
    _, ax1 = compare_graphs(G, H, seed=3)
    _, ax2 = compare_graphs(G, H, seed=3)
    off1 = ax1.collections[-1].get_offsets()
    off2 = ax2.collections[-1].get_offsets()
    assert (off1 == off2).all()


def test_compare_graphs_directed(plt):
    D = nx.DiGraph([(0, 1), (1, 2), (2, 0)])
    _, _, diff = compare_graphs(D, nx.DiGraph([(0, 1)]), seed=1, return_diff=True)
    assert diff["removed_edges"] == {(1, 2), (2, 0)}


def test_save_graph_comparison_writes_and_closes(plt, tmp_path):
    G = nx.cycle_graph(5)
    H = nx.path_graph(5)
    path = tmp_path / "cmp.png"
    before = len(plt.get_fignums())
    out = save_graph_comparison(G, H, path, seed=1, title="t", dpi=50)
    assert out == path
    assert path.exists() and path.stat().st_size > 0
    assert len(plt.get_fignums()) == before


def test_save_graph_comparison_return_diff(plt, tmp_path):
    """Regression: return_diff=True used to crash on tuple unpacking."""
    G = nx.cycle_graph(5)
    H = nx.path_graph(5)
    before = len(plt.get_fignums())
    out, diff = save_graph_comparison(
        G, H, tmp_path / "cmp.pdf", seed=1, return_diff=True
    )
    assert out.exists()
    assert diff["removed_edges"] == {(0, 4)}
    assert len(plt.get_fignums()) == before


def test_save_graph_comparison_keeps_caller_figure_open(plt, tmp_path):
    fig, ax = plt.subplots()
    save_graph_comparison(nx.path_graph(3), nx.path_graph(2), tmp_path / "x.png", ax=ax)
    assert plt.fignum_exists(fig.number)
