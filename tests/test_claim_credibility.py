import json

from arglib.ai import NoOpLLMClient, build_claim_credibility_hook, score_claims_with_llm
from arglib.core import ArgumentGraph, EvidenceCard


def test_claim_credibility_weighted_score():
    graph = ArgumentGraph.new()
    claim_id = graph.add_claim("Claim A")
    graph.add_evidence_card(
        EvidenceCard(
            id="e1",
            title="Doc 1",
            supporting_doc_id="doc1",
            excerpt="Supports strongly.",
            confidence=1.0,
        )
    )
    graph.add_evidence_card(
        EvidenceCard(
            id="e2",
            title="Doc 2",
            supporting_doc_id="doc2",
            excerpt="Contradicts.",
            confidence=0.2,
        )
    )
    graph.attach_evidence_card(claim_id, "e1")
    graph.attach_evidence_card(claim_id, "e2")

    response = json.dumps(
        {
            "claim_score": 0.1,
            "evidence": [
                {"evidence_id": "e1", "score": 1.0, "rationale": "direct support"},
                {"evidence_id": "e2", "score": -1.0, "rationale": "conflicts"},
            ],
            "summary": "Mixed evidence.",
        }
    )
    hook = build_claim_credibility_hook(NoOpLLMClient(response=response))

    results = score_claims_with_llm(
        graph, hook, unit_ids=[claim_id], use_llm_score=False
    )
    result = results[claim_id]

    expected = (1.0 * 1.0 + 0.2 * -1.0) / 1.2
    assert round(result.weighted_score, 6) == round(expected, 6)
    assert result.claim_score == result.weighted_score
    assert graph.units[claim_id].metadata["claim_credibility"] == result.claim_score


def test_claim_credibility_uses_llm_score_when_enabled():
    graph = ArgumentGraph.new()
    claim_id = graph.add_claim("Claim B")
    graph.add_evidence_card(
        EvidenceCard(
            id="e1",
            title="Doc 1",
            supporting_doc_id="doc1",
            excerpt="Supports strongly.",
            confidence=1.0,
        )
    )
    graph.attach_evidence_card(claim_id, "e1")

    response = json.dumps(
        {
            "claim_score": -0.4,
            "evidence": [
                {"evidence_id": "e1", "score": 0.9, "rationale": "mostly supports"}
            ],
            "summary": "LLM override.",
        }
    )
    hook = build_claim_credibility_hook(NoOpLLMClient(response=response))

    results = score_claims_with_llm(graph, hook, unit_ids=[claim_id])
    result = results[claim_id]

    assert result.claim_score == -0.4
    assert result.score_source == "llm"


def test_claim_credibility_no_evidence_defaults_to_zero():
    graph = ArgumentGraph.new()
    claim_id = graph.add_claim("Claim C")

    hook = build_claim_credibility_hook(NoOpLLMClient(response="{}"))
    results = score_claims_with_llm(graph, hook, unit_ids=[claim_id])
    result = results[claim_id]

    assert result.claim_score == 0.0
    assert result.weighted_score == 0.0


def test_claim_credibility_zero_trust_returns_zero():
    graph = ArgumentGraph.new()
    claim_id = graph.add_claim("Claim D")
    graph.add_evidence_card(
        EvidenceCard(
            id="e1",
            title="Doc 1",
            supporting_doc_id="doc1",
            excerpt="Supports strongly.",
            confidence=0.0,
        )
    )
    graph.attach_evidence_card(claim_id, "e1")

    response = json.dumps(
        {
            "claim_score": 0.9,
            "evidence": [
                {"evidence_id": "e1", "score": 1.0, "rationale": "support"},
            ],
            "summary": "Trust is zero.",
        }
    )
    hook = build_claim_credibility_hook(NoOpLLMClient(response=response))

    results = score_claims_with_llm(
        graph, hook, unit_ids=[claim_id], use_llm_score=False
    )
    result = results[claim_id]

    assert result.weighted_score == 0.0
