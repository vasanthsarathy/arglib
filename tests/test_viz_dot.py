from arglib.core import ArgumentGraph
from arglib.viz import to_dot


def test_to_dot_contains_nodes_and_edges():
    graph = ArgumentGraph.new()
    a = graph.add_claim("A")
    b = graph.add_claim("B")
    graph.add_attack(a, b)

    dot = to_dot(graph)

    assert "digraph ArgumentGraph" in dot
    assert f'"{a}"' in dot
    assert f'"{b}"' in dot
    assert "attack" in dot
