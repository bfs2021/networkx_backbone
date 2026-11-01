"""Tests for bipartite backbone extraction methods."""

import networkx as nx
import numpy as np
import pytest

from networkx_backbone import (
    backbone,
    backbone_from_projection,
    backbone_from_unweighted,
    backbone_from_weighted,
    bicm,
    bipartite_projection,
    fastball,
    fdsm,
    fixedcol,
    fixedfill,
    fixedrow,
    hyper_projection,
    probs_projection,
    sdsm,
    simple_projection,
    ycn_projection,
)


def test_davis_southern_women_fixture(davis_southern_women_graph, davis_women_nodes):
    assert nx.is_bipartite(davis_southern_women_graph)
    assert davis_southern_women_graph.number_of_nodes() == 32
    assert davis_southern_women_graph.number_of_edges() == 89
    assert len(davis_women_nodes) == 18


def test_simple_projection_weights(bipartite_graph):
    projection = simple_projection(bipartite_graph, agent_nodes=[1, 2, 3])
    assert projection.number_of_nodes() == 3
    assert projection.number_of_edges() == 3
    for _, _, data in projection.edges(data=True):
        assert data["weight"] == pytest.approx(1.0)


def test_hyper_projection_weights(bipartite_graph):
    projection = hyper_projection(bipartite_graph, agent_nodes=[1, 2, 3])
    assert projection.number_of_nodes() == 3
    assert projection.number_of_edges() == 3
    for _, _, data in projection.edges(data=True):
        assert data["weight"] == pytest.approx(0.5)


def test_probs_projection_weights(bipartite_graph):
    projection = probs_projection(bipartite_graph, agent_nodes=[1, 2, 3])
    assert projection.number_of_nodes() == 3
    assert projection.number_of_edges() == 3
    for _, _, data in projection.edges(data=True):
        assert data["weight"] == pytest.approx(0.25)


def test_ycn_projection_weights(bipartite_graph):
    projection = ycn_projection(bipartite_graph, agent_nodes=[1, 2, 3])
    assert projection.number_of_nodes() == 3
    assert projection.number_of_edges() == 3
    for _, _, data in projection.edges(data=True):
        assert data["weight"] == pytest.approx(1.0 / 12.0)


def test_probs_projection_directed():
    B = nx.Graph()
    B.add_edges_from([("a", "x"), ("a", "y"), ("b", "x"), ("c", "y")])
    projection = probs_projection(B, agent_nodes=["a", "b", "c"], directed=True)
    assert isinstance(projection, nx.DiGraph)
    assert projection["a"]["b"]["weight"] == pytest.approx(0.25)
    assert projection["b"]["a"]["weight"] == pytest.approx(0.5)


@pytest.mark.parametrize("method", ["simple", "hyper", "probs", "ycn"])
def test_bipartite_projection_dispatch_methods(bipartite_graph, method):
    projection = bipartite_projection(
        bipartite_graph, agent_nodes=[1, 2, 3], method=method
    )
    assert isinstance(projection, nx.Graph)
    assert set(projection.nodes()) == {1, 2, 3}


def test_bipartite_projection_dispatch_unknown_method_raises(bipartite_graph):
    with pytest.raises(ValueError):
        bipartite_projection(bipartite_graph, agent_nodes=[1, 2, 3], method="unknown")


class TestSDSM:
    def test_returns_graph(self, bipartite_graph):
        backbone = sdsm(bipartite_graph, agent_nodes=[1, 2, 3])
        assert isinstance(backbone, nx.Graph)

    def test_adds_pvalue_attribute(self, bipartite_graph):
        backbone = sdsm(bipartite_graph, agent_nodes=[1, 2, 3])
        for _u, _v, data in backbone.edges(data=True):
            assert "sdsm_pvalue" in data

    def test_agent_nodes_only(self, bipartite_graph):
        backbone = sdsm(bipartite_graph, agent_nodes=[1, 2, 3])
        for node in backbone.nodes():
            assert node in [1, 2, 3]

    def test_returns_full_projection(self, bipartite_graph):
        backbone = sdsm(bipartite_graph, agent_nodes=[1, 2, 3])
        assert backbone.number_of_nodes() == 3
        assert backbone.number_of_edges() == 3

    def test_non_bipartite_raises(self):
        G = nx.complete_graph(3)  # Not bipartite
        with pytest.raises(nx.NetworkXError):
            sdsm(G, agent_nodes=[0, 1])

    def test_invalid_agent_nodes_raises(self, bipartite_graph):
        with pytest.raises(nx.NetworkXError):
            sdsm(bipartite_graph, agent_nodes=[99])

    def test_davis_southern_women_graph(
        self, davis_southern_women_graph, davis_women_nodes
    ):
        backbone = sdsm(davis_southern_women_graph, agent_nodes=davis_women_nodes)
        assert set(backbone.nodes()).issubset(set(davis_women_nodes))
        assert (
            backbone.number_of_edges()
            == len(davis_women_nodes) * (len(davis_women_nodes) - 1) // 2
        )
        for _, _, data in backbone.edges(data=True):
            assert "sdsm_pvalue" in data

    def test_alpha_does_not_change_projection(self, bipartite_graph):
        low_alpha = sdsm(bipartite_graph, agent_nodes=[1, 2, 3], alpha=0.001)
        high_alpha = sdsm(bipartite_graph, agent_nodes=[1, 2, 3], alpha=0.9)
        assert set(low_alpha.edges()) == set(high_alpha.edges())

    def test_projection_weight_annotation(self, bipartite_graph):
        backbone = sdsm(bipartite_graph, agent_nodes=[1, 2, 3], projection="hyper")
        for _, _, data in backbone.edges(data=True):
            assert "sdsm_pvalue" in data
            assert "weight" in data

    def test_invalid_projection_raises(self, bipartite_graph):
        with pytest.raises(ValueError):
            sdsm(bipartite_graph, agent_nodes=[1, 2, 3], projection="invalid")


class TestFDSM:
    def test_returns_graph(self, bipartite_graph):
        backbone = fdsm(bipartite_graph, agent_nodes=[1, 2, 3], trials=50, seed=42)
        assert isinstance(backbone, nx.Graph)

    def test_adds_pvalue_attribute(self, bipartite_graph):
        backbone = fdsm(bipartite_graph, agent_nodes=[1, 2, 3], trials=50, seed=42)
        for _u, _v, data in backbone.edges(data=True):
            assert "fdsm_pvalue" in data

    def test_pvalues_in_range(self, bipartite_graph):
        backbone = fdsm(bipartite_graph, agent_nodes=[1, 2, 3], trials=50, seed=42)
        for _, _, data in backbone.edges(data=True):
            assert 0.0 <= data["fdsm_pvalue"] <= 1.0

    def test_returns_full_projection(self, bipartite_graph):
        backbone = fdsm(bipartite_graph, agent_nodes=[1, 2, 3], trials=50, seed=42)
        assert backbone.number_of_nodes() == 3
        assert backbone.number_of_edges() == 3

    def test_invalid_trials_raises(self, bipartite_graph):
        with pytest.raises(ValueError):
            fdsm(bipartite_graph, agent_nodes=[1, 2, 3], trials=0)

    def test_davis_southern_women_graph(
        self, davis_southern_women_graph, davis_women_nodes
    ):
        backbone = fdsm(
            davis_southern_women_graph,
            agent_nodes=davis_women_nodes,
            trials=50,
            seed=42,
        )
        assert set(backbone.nodes()).issubset(set(davis_women_nodes))
        assert (
            backbone.number_of_edges()
            == len(davis_women_nodes) * (len(davis_women_nodes) - 1) // 2
        )
        for _, _, data in backbone.edges(data=True):
            assert "fdsm_pvalue" in data

    def test_alpha_does_not_change_projection(self, bipartite_graph):
        low_alpha = fdsm(
            bipartite_graph, agent_nodes=[1, 2, 3], alpha=0.001, trials=50, seed=42
        )
        high_alpha = fdsm(
            bipartite_graph, agent_nodes=[1, 2, 3], alpha=0.9, trials=50, seed=42
        )
        assert set(low_alpha.edges()) == set(high_alpha.edges())

    def test_projection_weight_annotation(self, bipartite_graph):
        backbone = fdsm(
            bipartite_graph,
            agent_nodes=[1, 2, 3],
            trials=50,
            seed=42,
            projection="ycn",
        )
        for _, _, data in backbone.edges(data=True):
            assert "fdsm_pvalue" in data
            assert "weight" in data


class TestBipartiteUtilities:
    def test_bicm_returns_probability_matrix(
        self, davis_southern_women_graph, davis_women_nodes
    ):
        p, agents, artifacts = bicm(
            davis_southern_women_graph, davis_women_nodes, return_labels=True
        )
        assert p.shape == (len(agents), len(artifacts))
        assert np.all(p >= 0.0)
        assert np.all(p <= 1.0)

    def test_fastball_preserves_marginals(self):
        m = np.array(
            [
                [1, 0, 1, 0, 0],
                [0, 1, 0, 1, 0],
                [1, 1, 0, 0, 1],
                [0, 0, 1, 1, 1],
            ],
            dtype=int,
        )
        randomized = fastball(m, n_swaps=50, seed=42)
        assert randomized.shape == m.shape
        assert np.array_equal(randomized.sum(axis=1), m.sum(axis=1))
        assert np.array_equal(randomized.sum(axis=0), m.sum(axis=0))


@pytest.mark.parametrize(
    ("method", "attr"),
    [
        (fixedfill, "fixedfill_pvalue"),
        (fixedrow, "fixedrow_pvalue"),
        (fixedcol, "fixedcol_pvalue"),
    ],
)
def test_fixed_projection_methods(
    davis_southern_women_graph, davis_women_nodes, method, attr
):
    backbone_graph = method(davis_southern_women_graph, davis_women_nodes, alpha=0.5)
    assert set(backbone_graph.nodes()).issubset(set(davis_women_nodes))
    for _, _, data in backbone_graph.edges(data=True):
        assert attr in data
        assert 0.0 <= data[attr] <= 1.0


def test_backbone_from_projection_dispatch(
    davis_southern_women_graph, davis_women_nodes
):
    backbone_graph = backbone_from_projection(
        davis_southern_women_graph,
        davis_women_nodes,
        method="fixedrow",
        alpha=0.5,
        projection="probs",
    )
    assert set(backbone_graph.nodes()).issubset(set(davis_women_nodes))


def test_backbone_from_projection_sdsm_with_custom_projection(
    davis_southern_women_graph, davis_women_nodes
):
    backbone_graph = backbone_from_projection(
        davis_southern_women_graph,
        davis_women_nodes,
        method="sdsm",
        alpha=0.5,
        projection="ycn",
    )
    scored = sdsm(davis_southern_women_graph, davis_women_nodes, projection="ycn")
    expected = {
        frozenset((u, v)) for u, v, p in scored.edges(data="sdsm_pvalue") if p < 0.5
    }
    assert backbone_graph.number_of_nodes() == len(davis_women_nodes)
    assert {frozenset(e) for e in backbone_graph.edges()} == expected
    assert expected
    for _, _, data in backbone_graph.edges(data=True):
        assert data["sdsm_pvalue"] < 0.5
        assert "weight" in data


def test_backbone_from_weighted_dispatch(weighted_triangle):
    backbone_graph = backbone_from_weighted(
        weighted_triangle, method="disparity", alpha=1.0
    )
    assert backbone_graph.number_of_nodes() == weighted_triangle.number_of_nodes()
    with pytest.raises(ValueError):
        backbone_from_weighted(weighted_triangle, method="global")


def test_backbone_from_weighted_multigraph_auto_collapse():
    G = nx.MultiGraph()
    G.add_edge(0, 1, edge_type="a")
    G.add_edge(0, 1, edge_type="b")
    G.add_edge(0, 1, edge_type="b")
    G.add_edge(1, 2, edge_type="a")
    backbone_graph = backbone_from_weighted(
        G, method="global", threshold=2, edge_type_attr="edge_type"
    )
    assert backbone_graph.number_of_edges() == 1
    assert backbone_graph.has_edge(0, 1)


def test_backbone_from_weighted_multidigraph_auto_collapse():
    G = nx.MultiDiGraph()
    G.add_edge(0, 1)
    G.add_edge(0, 1)
    G.add_edge(1, 0)
    backbone_graph = backbone_from_weighted(G, method="global", threshold=2)
    assert isinstance(backbone_graph, nx.DiGraph)
    assert list(backbone_graph.edges()) == [(0, 1)]


def test_backbone_from_unweighted_dispatch(karate):
    backbone_graph = backbone_from_unweighted(karate, method="lspar", s=0.5)
    assert backbone_graph.number_of_nodes() == karate.number_of_nodes()


def test_backbone_dispatch(
    davis_southern_women_graph, davis_women_nodes, weighted_triangle
):
    proj_backbone = backbone(
        davis_southern_women_graph,
        method="sdsm",
        agent_nodes=davis_women_nodes,
        alpha=0.5,
    )
    weighted_backbone = backbone(weighted_triangle, method="disparity", alpha=1.0)
    assert set(proj_backbone.nodes()).issubset(set(davis_women_nodes))
    assert weighted_backbone.number_of_nodes() == weighted_triangle.number_of_nodes()

    with pytest.raises(ValueError):
        backbone(weighted_triangle, method="not_a_method")


class TestSignedProjections:
    def test_sdsm_signed_adds_sign(self, davis_southern_women_graph, davis_women_nodes):
        H = sdsm(davis_southern_women_graph, agent_nodes=davis_women_nodes, signed=True)
        assert all(
            "sign" in d and d["sign"] in (-1, 1) for _, _, d in H.edges(data=True)
        )
        assert {d["sign"] for _, _, d in H.edges(data=True)} == {-1, 1}
        assert all(0.0 <= d["sdsm_pvalue"] <= 1.0 for _, _, d in H.edges(data=True))

    def test_sdsm_unsigned_has_no_sign(
        self, davis_southern_women_graph, davis_women_nodes
    ):
        H = sdsm(davis_southern_women_graph, agent_nodes=davis_women_nodes)
        assert all("sign" not in d for _, _, d in H.edges(data=True))

    def test_fdsm_signed_deterministic(
        self, davis_southern_women_graph, davis_women_nodes
    ):
        kw = {
            "agent_nodes": davis_women_nodes,
            "trials": 200,
            "seed": 7,
            "signed": True,
        }
        a = fdsm(davis_southern_women_graph, **kw)
        b = fdsm(davis_southern_women_graph, **kw)
        assert all("sign" in d for _, _, d in a.edges(data=True))
        assert {(u, v): d["sign"] for u, v, d in a.edges(data=True)} == {
            (u, v): d["sign"] for u, v, d in b.edges(data=True)
        }

    @pytest.mark.parametrize("model", [fixedrow, fixedcol, fixedfill])
    def test_fixed_models_signed(
        self, model, davis_southern_women_graph, davis_women_nodes
    ):
        H = model(davis_southern_women_graph, davis_women_nodes, alpha=0.3, signed=True)
        assert H.number_of_edges() > 0
        assert all(
            "sign" in d and d["sign"] in (-1, 1) for _, _, d in H.edges(data=True)
        )

    def test_backbone_from_projection_signed_passthrough(
        self, davis_southern_women_graph, davis_women_nodes
    ):
        H = backbone_from_projection(
            davis_southern_women_graph,
            davis_women_nodes,
            method="fixedrow",
            alpha=0.3,
            signed=True,
        )
        assert all("sign" in d for _, _, d in H.edges(data=True))


class TestSDSMEdgeConstraints:
    def test_constraints_change_null(
        self, davis_southern_women_graph, davis_women_nodes
    ):
        B = davis_southern_women_graph
        women = davis_women_nodes
        artifacts = sorted(
            (n for n, d in B.nodes(data=True) if d["bipartite"] == 1), key=str
        )
        present = next((women[0], f) for f in artifacts if B.has_edge(women[0], f))
        absent = next((women[0], f) for f in artifacts if not B.has_edge(women[0], f))

        base = sdsm(B, agent_nodes=women)
        prohibited = sdsm(B, agent_nodes=women, prohibited=[absent])
        required = sdsm(B, agent_nodes=women, required=[present])

        diff_pro = sum(
            1
            for u, v, d in prohibited.edges(data=True)
            if abs(d["sdsm_pvalue"] - base[u][v]["sdsm_pvalue"]) > 1e-9
        )
        diff_req = sum(
            1
            for u, v, d in required.edges(data=True)
            if abs(d["sdsm_pvalue"] - base[u][v]["sdsm_pvalue"]) > 1e-9
        )
        assert diff_pro > 0 and diff_req > 0

    def test_unknown_constraint_nodes_raise(
        self, davis_southern_women_graph, davis_women_nodes
    ):
        B = davis_southern_women_graph
        with pytest.raises(nx.NetworkXError):
            sdsm(B, agent_nodes=davis_women_nodes, prohibited=[("missing", "absent")])

    def test_constraints_compose_with_signed(
        self, davis_southern_women_graph, davis_women_nodes
    ):
        artifact = next(n for n in davis_southern_women_graph[davis_women_nodes[0]])
        H = sdsm(
            davis_southern_women_graph,
            agent_nodes=davis_women_nodes,
            signed=True,
            required=[(davis_women_nodes[0], artifact)],
        )
        assert all("sign" in d for _, _, d in H.edges(data=True))
