"""Shared test fixtures for networkx_backbone tests."""

import networkx as nx
import pytest


@pytest.fixture
def weighted_triangle():
    """Simple weighted triangle graph: 0-1 (3.0), 1-2 (1.0), 0-2 (2.0)."""
    G = nx.Graph()
    G.add_weighted_edges_from([(0, 1, 3.0), (1, 2, 1.0), (0, 2, 2.0)])
    return G


@pytest.fixture
def karate():
    """Zachary's karate club graph (unweighted)."""
    return nx.karate_club_graph()


@pytest.fixture
def complete4():
    """Complete graph on 4 nodes (unweighted)."""
    return nx.complete_graph(4)


@pytest.fixture
def bipartite_graph():
    """Simple bipartite graph with agents [1,2,3] and artifacts ['a','b','c']."""
    B = nx.Graph()
    B.add_edges_from(
        [
            (1, "a"),
            (1, "b"),
            (2, "a"),
            (2, "c"),
            (3, "b"),
            (3, "c"),
        ]
    )
    return B


@pytest.fixture
def davis_southern_women_graph():
    """Davis Southern Women bipartite attendance network."""
    return nx.davis_southern_women_graph()


@pytest.fixture
def davis_women_nodes(davis_southern_women_graph):
    """Women partition from the Davis Southern Women graph."""
    return [
        n
        for n, data in davis_southern_women_graph.nodes(data=True)
        if data.get("bipartite") == 0
    ]


@pytest.fixture
def les_miserables_weighted():
    """Weighted co-appearance graph from Victor Hugo's *Les Miserables*."""
    return nx.les_miserables_graph()


@pytest.fixture
def les_miserables_unweighted(les_miserables_weighted):
    """Unweighted version of Les Miserables preserving nodes but dropping edge attrs."""
    G = nx.Graph()
    G.add_nodes_from(les_miserables_weighted.nodes(data=True))
    G.add_edges_from(les_miserables_weighted.edges())
    return G
