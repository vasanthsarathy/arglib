from arglib.core import ArgumentGraph
from arglib.critique import apply_gate_actions, detect_patterns


def test_pattern_detection_basic():
    graph = ArgumentGraph.new()
    a = graph.add_claim("A")
    b = graph.add_claim("B")
    graph.add_support(a, b)
    graph.add_attack(a, a)

    matches = detect_patterns(graph)
    pattern_ids = {match.pattern_id for match in matches}

    assert "self_attack" in pattern_ids
    assert "unsupported_conclusion" in pattern_ids
    assert "unstated_warrant" in pattern_ids


def test_circular_reasoning_detected_and_disabled():
    graph = ArgumentGraph.new()
    a = graph.add_claim("A")
    b = graph.add_claim("B")
    graph.add_support(a, b)
    graph.add_support(b, a)

    matches = detect_patterns(graph)
    apply_gate_actions(graph, matches)

    disabled = [
        rel.metadata.get("gate_disabled", False) for rel in graph.relations
    ]
    assert any(disabled)
