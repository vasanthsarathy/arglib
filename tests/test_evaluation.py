from arglib.ai import HeuristicEvaluator, score_evidence, validate_edges
from arglib.core import ArgumentGraph


def test_score_evidence_sets_strength():
    graph = ArgumentGraph.new()
    node_id = graph.add_claim("Dogs are friendly animals.")
    graph.attach_evidence(
        node_id,
        evidence_id="e1",
        source={"text": "Dogs are often friendly with humans."},
        stance="neutral",
    )

    results = score_evidence(graph, HeuristicEvaluator())

    assert results
    assert graph.units[node_id].evidence[0].strength is not None


def test_validate_edges_sets_metadata():
    graph = ArgumentGraph.new()
    a = graph.add_claim("Cats are independent.")
    b = graph.add_claim("Cats are independent pets.")
    graph.add_support(a, b)

    results = validate_edges(graph, HeuristicEvaluator())

    assert results
    assert "confidence" in graph.relations[0].metadata
