from arglib.core import ArgumentGraph
from arglib.reasoning import Reasoner


def test_reasoner_grounded_extension():
    graph = ArgumentGraph.new()
    a = graph.add_claim("A")
    b = graph.add_claim("B")
    graph.add_attack(a, b)

    reasoner = Reasoner(graph)
    results = reasoner.run(["grounded_extension", "grounded_labeling"])

    assert results["grounded_extension"] == [a]
    assert results["grounded_labeling"][a] == "in"
    assert results["grounded_labeling"][b] == "out"
