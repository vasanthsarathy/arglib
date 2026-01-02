from arglib.core import ArgumentGraph
from arglib.reasoning import Reasoner


def test_reasoner_credibility_and_gate_scores():
    graph = ArgumentGraph.new()
    a = graph.add_claim("A")
    b = graph.add_claim("B")
    w1 = graph.add_warrant("A is relevant to B", type="fact")
    graph.add_support(a, b, warrant_ids=[w1], gate_mode="OR")

    reasoner = Reasoner(graph)
    results = reasoner.run(["credibility_propagation", "gate_scores"])

    assert "final_scores" in results["credibility_propagation"]
    assert results["gate_scores"]
