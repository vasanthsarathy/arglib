from arglib.core import ArgumentGraph, EvidenceCard
from arglib.reasoning import compute_credibility


def test_credibility_propagation_support():
    graph = ArgumentGraph.new()
    a = graph.add_claim("A")
    b = graph.add_claim("B")
    graph.add_support(a, b)

    graph.add_evidence_card(
        EvidenceCard(
            id="e1",
            title="Study",
            supporting_doc_id="doc1",
            excerpt="...",
            confidence=1.0,
        )
    )
    graph.attach_evidence_card(a, "e1")

    result = compute_credibility(graph, lambda_=0.5)

    assert result.initial_evidence[a] == 1.0
    assert result.final_scores[b] > 0


def test_credibility_propagation_attack():
    graph = ArgumentGraph.new()
    a = graph.add_claim("A")
    b = graph.add_claim("B")
    graph.add_attack(a, b)

    graph.add_evidence_card(
        EvidenceCard(
            id="e1",
            title="Counter",
            supporting_doc_id="doc1",
            excerpt="...",
            confidence=1.0,
        )
    )
    graph.attach_evidence_card(a, "e1")

    result = compute_credibility(graph, lambda_=0.5)

    assert result.final_scores[b] < 0
