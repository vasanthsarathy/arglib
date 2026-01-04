from arglib.core import ArgumentGraph
from arglib.reasoning import compute_credibility, explain_credibility


def test_explain_credibility_returns_incoming_influences():
    graph = ArgumentGraph.new()
    a = graph.add_claim("A")
    b = graph.add_claim("B")
    w1 = graph.add_warrant("A warrants B", type="fact")
    graph.add_support(a, b, warrant_ids=[w1], gate_mode="OR")

    result = compute_credibility(graph)
    explanation = explain_credibility(graph, result)

    assert b in explanation
    assert explanation[b]["incoming"]
