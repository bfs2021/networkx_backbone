"""Tests for hypergraph ingestion / interoperability adapters."""

import importlib
import io
import json
import sys
import types

import networkx as nx
import numpy as np
import pytest

import networkx_backbone as nb

HYPEREDGES = [(1, 2, 3), (2, 3, 4), (5, 6)]
EXPECTED = {frozenset(e) for e in HYPEREDGES}


# ---------------------------------------------------------------------------
# Incidence bipartite graph
# ---------------------------------------------------------------------------


def test_hypergraph_to_bipartite_structure():
    B, nodes = nb.hypergraph_to_bipartite(HYPEREDGES)
    assert nx.is_bipartite(B)
    assert nodes == [1, 2, 3, 4, 5, 6]
    node_part = {n for n, d in B.nodes(data=True) if d["bipartite"] == 0}
    edge_part = {n for n, d in B.nodes(data=True) if d["bipartite"] == 1}
    assert node_part == set(nodes)
    assert len(edge_part) == 3
    # Each hyperedge node records its members.
    assert any(B.nodes[e]["members"] == frozenset({1, 2, 3}) for e in edge_part)


def test_hypergraph_to_bipartite_tuple_edge_labels():
    B, _ = nb.hypergraph_to_bipartite(HYPEREDGES)
    assert B.nodes[("he", 0)]["members"] == frozenset({1, 2, 3})
    assert B.nodes[("he", 2)]["bipartite"] == 1
    B2, _ = nb.hypergraph_to_bipartite(HYPEREDGES, edge_prefix="edge")
    assert ("edge", 1) in B2


def test_hypergraph_to_bipartite_string_label_no_collision():
    # Regression: a node called "he0" used to merge with hyperedge 0's node,
    # creating a self-loop and a non-bipartite graph.
    B, nodes = nb.hypergraph_to_bipartite([("he0", "b"), ("b", "c")])
    assert nx.number_of_selfloops(B) == 0
    assert nx.is_bipartite(B)
    assert nodes == ["b", "c", "he0"]
    assert B.nodes["he0"]["bipartite"] == 0
    assert set(B[("he", 0)]) == {"he0", "b"}
    assert B.number_of_edges() == 4


def test_hypergraph_to_bipartite_tuple_label_collision_raises():
    with pytest.raises(ValueError, match="edge_prefix"):
        nb.hypergraph_to_bipartite([(("he", 0), "b")])
    B, _ = nb.hypergraph_to_bipartite([(("he", 0), "b")], edge_prefix="x")
    assert nx.is_bipartite(B) and nx.number_of_selfloops(B) == 0


def test_hypergraph_to_bipartite_mixed_labels_and_empty_edges():
    B, nodes = nb.hypergraph_to_bipartite([(1, "a"), (), ("a", (2, 3))])
    assert set(nodes) == {1, "a", (2, 3)}
    assert nodes == sorted(nodes, key=repr)
    assert sum(1 for _, d in B.nodes(data=True) if d["bipartite"] == 1) == 2


def test_hypergraph_to_bipartite_feeds_sdsm():
    B, nodes = nb.hypergraph_to_bipartite(HYPEREDGES)
    scored = nb.sdsm(B, agent_nodes=nodes)
    assert all("sdsm_pvalue" in d for _, _, d in scored.edges(data=True))


# ---------------------------------------------------------------------------
# HIF
# ---------------------------------------------------------------------------


def test_hif_roundtrip_dict():
    hif = nb.write_hif(HYPEREDGES)
    assert hif["network-type"] == "undirected"
    assert set(nb.read_hif(hif)) == EXPECTED


def test_hif_roundtrip_with_weights():
    hif = nb.write_hif([(1, 2), (3, 4)], weights=[5, 2])
    edges, weights = nb.read_hif(hif, return_weights=True)
    assert dict(zip((tuple(sorted(e)) for e in edges), weights)) == {
        (1, 2): 5,
        (3, 4): 2,
    }


def test_hif_roundtrip_file(tmp_path):
    path = tmp_path / "h.json"
    nb.write_hif(HYPEREDGES, path)
    assert set(nb.read_hif(path)) == EXPECTED


def test_write_hif_skips_empty_hyperedges():
    edges, weights = nb.read_hif(
        nb.write_hif([(1, 2), (), (2, 3)], weights=[4, 9, 5]), return_weights=True
    )
    assert edges == [frozenset({1, 2}), frozenset({2, 3})]
    assert weights == [4, 5]


def test_write_hif_weight_length_checked():
    with pytest.raises(ValueError):
        nb.write_hif([(1, 2), (3, 4)], weights=[1])


def test_read_hif_bad_source_raises():
    with pytest.raises(TypeError):
        nb.read_hif(12345)


def test_hif_numpy_weights_and_ids_to_file(tmp_path):
    # Regression: numpy scalars are not JSON-serialisable.
    path = tmp_path / "np.json"
    edges = [np.array([1, 2]), np.array([2, 3, 4])]
    nb.write_hif(edges, path, weights=np.array([3, 1]))
    got, weights = nb.read_hif(path, return_weights=True)
    assert got == [frozenset({1, 2}), frozenset({2, 3, 4})]
    assert weights == [3, 1]
    assert all(type(w) is int for w in weights)
    assert all(type(v) is int for e in got for v in e)

    hif = nb.write_hif(edges, weights=np.array([1.5, 2.0]))
    json.dumps(hif)  # the returned dict is JSON-serialisable too
    assert [rec["weight"] for rec in hif["edges"]] == [1.5, 2.0]


def test_hif_file_like_roundtrip():
    buf = io.StringIO()
    assert nb.write_hif(HYPEREDGES, buf, weights=[1, 2, 3]) is buf
    buf.seek(0)
    edges, weights = nb.read_hif(buf, return_weights=True)
    assert set(edges) == EXPECTED and weights == [1, 2, 3]


def test_hif_tuple_node_ids_roundtrip(tmp_path):
    # Regression: tuple labels came back as (unhashable) lists after JSON.
    H = [((0, 1), (0, 2)), ((0, 2), ("a", (1, 2)))]
    path = tmp_path / "tuples.hif"
    nb.write_hif(H, str(path))
    assert nb.read_hif(path) == [frozenset(e) for e in H]
    # JSON-decoded dicts behave the same as files.
    assert nb.read_hif(json.loads(json.dumps(nb.write_hif(H)))) == [
        frozenset(e) for e in H
    ]


def test_hif_mixed_int_and_str_labels_stay_distinct():
    hif = json.loads(json.dumps(nb.write_hif([(1, "1")])))
    assert nb.read_hif(hif) == [frozenset({1, "1"})]


def test_read_hif_directed_raises():
    hif = {
        "network-type": "directed",
        "incidences": [
            {"edge": 0, "node": "a", "direction": "head"},
            {"edge": 0, "node": "b", "direction": "tail"},
        ],
    }
    with pytest.raises(nx.NetworkXNotImplemented):
        nb.read_hif(hif)


def test_write_hif_directed_or_unknown_type_raises():
    with pytest.raises(nx.NetworkXNotImplemented):
        nb.write_hif(HYPEREDGES, network_type="directed")
    with pytest.raises(ValueError, match="network-type"):
        nb.write_hif(HYPEREDGES, network_type="hyper")
    with pytest.raises(ValueError, match="network-type"):
        nb.read_hif({"network-type": "bogus", "incidences": []})


def test_read_hif_spec_variants():
    spec = {
        "network-type": "asc",
        "incidences": [
            {"edge": "e1", "node": "a"},
            {"edge": "e1", "node": "b"},
            {"edge": "e2", "node": "b"},
            {"edge": "e3", "node": "c"},
        ],
        "edges": [
            {"edge": "e1", "attrs": {"weight": 4}},
            {"edge": "e2", "weight": 2.5},
            {"edge": "e3", "attrs": {}},
        ],
    }
    edges, weights = nb.read_hif(spec, return_weights=True)
    assert edges == [frozenset("ab"), frozenset("b"), frozenset("c")]
    assert weights == [4, 2.5, 1]
    # Missing network-type defaults to undirected.
    assert nb.read_hif({"incidences": [{"edge": 0, "node": 1}]}) == [frozenset({1})]
    assert nb.write_hif([(1, 2)], network_type="asc")["network-type"] == "asc"


# ---------------------------------------------------------------------------
# Cross-library interoperability (run when a library is available)
# ---------------------------------------------------------------------------


def test_hif_crosscheck_with_xgi(tmp_path):
    xgi = pytest.importorskip("xgi")
    path = tmp_path / "h.json"
    # xgi reads HIF we wrote.
    nb.write_hif(HYPEREDGES, path)
    Hx = xgi.read_hif(path)
    assert {frozenset(m) for m in Hx.edges.members()} == EXPECTED
    # We read HIF xgi wrote.
    xgi.write_hif(xgi.Hypergraph([list(e) for e in HYPEREDGES]), path)
    assert set(nb.read_hif(path)) == EXPECTED


ADAPTERS = [
    ("xgi", nb.to_xgi, nb.from_xgi),
    ("hypernetx", nb.to_hypernetx, nb.from_hypernetx),
    ("hypergraphx", nb.to_hypergraphx, nb.from_hypergraphx),
    ("HAT", nb.to_hat, nb.from_hat),
]


@pytest.mark.parametrize("module,to_fn,from_fn", ADAPTERS, ids=[a[0] for a in ADAPTERS])
def test_adapter_roundtrip_or_importerror(module, to_fn, from_fn):
    try:
        importlib.import_module(module)
    except ImportError:
        # Absent (or broken) optional dependency raises a helpful ImportError.
        with pytest.raises(ImportError, match=module):
            to_fn(HYPEREDGES)
        return

    obj = to_fn(HYPEREDGES)
    recovered = from_fn(obj)
    if module == "HAT":
        # HAT is positional: labels become indices, so compare the order multiset.
        assert sorted(len(e) for e in recovered) == sorted(
            len(set(e)) for e in HYPEREDGES
        )
    else:
        assert set(recovered) == EXPECTED


# ---------------------------------------------------------------------------
# Adapter logic, exercised with lightweight stand-in modules
# ---------------------------------------------------------------------------


def _install(monkeypatch, name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


INPUT = [(1, 2, 3), (), (3, 4), (1, 2, 3, 3)]  # empty edge dropped, dup kept
CLEAN = [frozenset({1, 2, 3}), frozenset({3, 4}), frozenset({1, 2, 3})]


def test_xgi_adapter_with_stub(monkeypatch):
    class Hypergraph:
        def __init__(self, members):
            self._members = [set(m) for m in members]
            outer = self

            class _Edges:
                def members(self):
                    return list(outer._members)

            self.edges = _Edges()

    _install(monkeypatch, "xgi", Hypergraph=Hypergraph)
    H = nb.to_xgi(INPUT)
    assert isinstance(H, Hypergraph)
    assert H._members == [set(e) for e in CLEAN]
    assert nb.from_xgi(H) == CLEAN


def test_hypernetx_adapter_with_stub(monkeypatch):
    class Hypergraph:
        def __init__(self, setsystem):
            self.incidence_dict = dict(setsystem)

    _install(monkeypatch, "hypernetx", Hypergraph=Hypergraph)
    H = nb.to_hypernetx(INPUT)
    assert list(H.incidence_dict) == [0, 1, 2]
    assert nb.from_hypernetx(H) == CLEAN


def test_hypergraphx_adapter_with_stub(monkeypatch):
    class Hypergraph:
        def __init__(self, edge_list=None):
            self._edges = list(edge_list)

        def get_edges(self):
            return list(self._edges)

    _install(monkeypatch, "hypergraphx", Hypergraph=Hypergraph)
    H = nb.to_hypergraphx(INPUT)
    assert all(isinstance(e, tuple) for e in H._edges)
    assert nb.from_hypergraphx(H) == CLEAN


def test_hat_adapter_with_stub(monkeypatch):
    class Hypergraph:
        def __init__(self, incidence_matrix=None):
            self.incidence_matrix = incidence_matrix

    _install(monkeypatch, "HAT", Hypergraph=Hypergraph)
    H = nb.to_hat([("b", "a"), (), ("c", "b", "d"), ("a", "d")])
    M = H.incidence_matrix
    # Rows follow sorted labels a, b, c, d; empty edge dropped.
    assert M.tolist() == [[1, 0, 1], [1, 1, 0], [0, 1, 0], [0, 1, 1]]
    assert nb.from_hat(H) == [
        frozenset({0, 1}),
        frozenset({1, 2, 3}),
        frozenset({0, 3}),
    ]


@pytest.mark.parametrize("attr", ["IM", "incidence_matrix", "incidenceMatrix"])
def test_from_hat_incidence_matrix_variants(attr):
    M = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 0]])

    class Plain:
        pass

    class Method:
        pass

    plain = Plain()
    setattr(plain, attr, M)
    method = Method()
    setattr(method, attr, lambda: M)
    expected = [frozenset({0, 1}), frozenset({1, 2})]  # all-zero column dropped
    assert nb.from_hat(plain) == expected
    assert nb.from_hat(method) == expected


def test_from_hat_sparse_and_missing_matrix():
    sparse = pytest.importorskip("scipy.sparse")

    class H:
        IM = sparse.csr_matrix(np.array([[1, 0], [1, 1], [0, 1]]))

    assert nb.from_hat(H()) == [frozenset({0, 1}), frozenset({1, 2})]
    with pytest.raises(AttributeError, match="incidence matrix"):
        nb.from_hat(object())


@pytest.mark.parametrize(
    "module,to_fn,pip_name",
    [
        ("xgi", nb.to_xgi, "xgi"),
        ("hypernetx", nb.to_hypernetx, "hypernetx"),
        ("hypergraphx", nb.to_hypergraphx, "hypergraphx"),
        ("HAT", nb.to_hat, "HypergraphAnalysisToolbox"),
    ],
)
def test_adapter_missing_library_importerror(monkeypatch, module, to_fn, pip_name):
    monkeypatch.setitem(sys.modules, module, None)  # forces ImportError
    with pytest.raises(ImportError, match=f"pip install {pip_name}"):
        to_fn(HYPEREDGES)
