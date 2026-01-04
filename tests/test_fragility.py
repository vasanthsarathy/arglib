from arglib.core import ArgumentGraph
from arglib.critique import analyze_warrant_fragility


def test_warrant_fragility_identifies_critical_warrant():
    graph = ArgumentGraph.new()
    a = graph.add_claim("A")
    b = graph.add_claim("B")
    w1 = graph.add_warrant("A warrants B", type="fact")
    graph.add_support(a, b, warrant_ids=[w1], gate_mode="OR")

    fragility = analyze_warrant_fragility(graph)

    assert fragility
    assert fragility[0].critical_warrants == [w1]
