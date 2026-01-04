from arglib.core import ArgumentGraph, EvidenceCard, SupportingDocument
from arglib.reasoning import compute_credibility


def test_credibility_propagation_support():
    graph = ArgumentGraph.new()
    a = graph.add_claim("A")
    b = graph.add_claim("B")
    w1 = graph.add_warrant("A warrants B", type="fact")
    graph.add_support(a, b, warrant_ids=[w1], gate_mode="OR")

    graph.add_evidence_card(
        EvidenceCard(
            id="e1",
            title="Study",
            supporting_doc_id="doc1",
            excerpt="...",
            confidence=1.0,
        )
    )
    graph.add_supporting_document(
        SupportingDocument(id="doc1", name="Doc", type="pdf", url="file://doc1")
    )
    graph.attach_evidence_card(a, "e1")
    graph.attach_evidence_to_warrant(
        w1, "e1", source={"evidence_card_id": "e1"}, stance="supports"
    )

    result = compute_credibility(graph, lambda_=0.5)

    assert result.initial_evidence[a] > 0
    assert result.final_scores[b] > 0.5


def test_credibility_propagation_attack():
    graph = ArgumentGraph.new()
    a = graph.add_claim("A")
    b = graph.add_claim("B")
    w1 = graph.add_warrant("A attacks B", type="fact")
    graph.add_attack(a, b, warrant_ids=[w1], gate_mode="OR")

    graph.add_evidence_card(
        EvidenceCard(
            id="e1",
            title="Counter",
            supporting_doc_id="doc1",
            excerpt="...",
            confidence=1.0,
        )
    )
    graph.add_supporting_document(
        SupportingDocument(id="doc1", name="Doc", type="pdf", url="file://doc1")
    )
    graph.attach_evidence_card(a, "e1")
    graph.attach_evidence_to_warrant(
        w1, "e1", source={"evidence_card_id": "e1"}, stance="supports"
    )

    result = compute_credibility(graph, lambda_=0.5)

    assert result.final_scores[b] < 0.5


def test_axiom_claims_and_warrants_seed_scores():
    graph = ArgumentGraph.new()
    a = graph.add_claim("A", is_axiom=True, score=1.0)
    b = graph.add_claim("B")
    w1 = graph.add_warrant("A supports B", is_axiom=True, score=1.0)
    graph.add_support(a, b, warrant_ids=[w1], gate_mode="OR")

    result = compute_credibility(graph, lambda_=0.5)

    assert result.final_scores[a] > 0.5
    assert result.final_scores[b] > 0.5


def test_ignore_influence_keeps_axiom_base():
    graph = ArgumentGraph.new()
    a = graph.add_claim("A", is_axiom=True, score=1.0)
    b = graph.add_claim(
        "B",
        is_axiom=True,
        score=-1.0,
        ignore_influence=True,
    )
    w1 = graph.add_warrant("A supports B", is_axiom=True, score=1.0)
    graph.add_support(a, b, warrant_ids=[w1], gate_mode="OR")

    result = compute_credibility(graph, lambda_=0.5)

    assert result.final_scores[b] < 0.5
