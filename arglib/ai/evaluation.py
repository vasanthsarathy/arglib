"""Deterministic evaluation helpers for edges and evidence."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from arglib.core import ArgumentGraph, EvidenceItem


@dataclass
class EvidenceEvaluation:
    node_id: str
    evidence_id: str
    evaluation: str  # supports, attacks, neutral
    reasoning: str
    confidence: float


@dataclass
class EdgeValidationResult:
    src: str
    dst: str
    evaluation: str  # support, attack, neutral
    reasoning: str
    confidence: float


class HeuristicEvaluator:
    """Simple deterministic evaluator for offline workflows."""

    def evaluate_evidence(
        self, claim_text: str, evidence_text: str
    ) -> EvidenceEvaluation:
        overlap = _token_overlap(claim_text, evidence_text)
        if overlap >= 0.35:
            return EvidenceEvaluation(
                node_id="",
                evidence_id="",
                evaluation="supports",
                reasoning="High lexical overlap suggests support.",
                confidence=min(1.0, overlap),
            )
        if overlap <= 0.05:
            return EvidenceEvaluation(
                node_id="",
                evidence_id="",
                evaluation="neutral",
                reasoning="Low lexical overlap suggests neutrality.",
                confidence=0.0,
            )
        return EvidenceEvaluation(
            node_id="",
            evidence_id="",
            evaluation="supports",
            reasoning="Moderate overlap suggests weak support.",
            confidence=overlap * 0.5,
        )

    def validate_edge(self, source_text: str, target_text: str) -> EdgeValidationResult:
        overlap = _token_overlap(source_text, target_text)
        if overlap >= 0.4:
            return EdgeValidationResult(
                src="",
                dst="",
                evaluation="support",
                reasoning="Claims share key terms, indicating support.",
                confidence=min(1.0, overlap),
            )
        if overlap <= 0.05:
            return EdgeValidationResult(
                src="",
                dst="",
                evaluation="neutral",
                reasoning="Little overlap; relation is unclear.",
                confidence=0.0,
            )
        return EdgeValidationResult(
            src="",
            dst="",
            evaluation="support",
            reasoning="Some overlap suggests weak support.",
            confidence=overlap * 0.5,
        )


def score_evidence(
    graph: ArgumentGraph, evaluator: HeuristicEvaluator | None = None
) -> list[EvidenceEvaluation]:
    evaluator = evaluator or HeuristicEvaluator()
    results: list[EvidenceEvaluation] = []
    for node_id, unit in graph.units.items():
        for item in unit.evidence:
            evidence_text = _evidence_text(item)
            evaluation = evaluator.evaluate_evidence(unit.text, evidence_text)
            evaluation.node_id = node_id
            evaluation.evidence_id = item.id
            item.strength = evaluation.confidence
            item.stance = _normalize_stance(evaluation.evaluation)
            results.append(evaluation)
    return results


def validate_edges(
    graph: ArgumentGraph, evaluator: HeuristicEvaluator | None = None
) -> list[EdgeValidationResult]:
    evaluator = evaluator or HeuristicEvaluator()
    results: list[EdgeValidationResult] = []
    for relation in graph.relations:
        source = graph.units.get(relation.src)
        target = graph.units.get(relation.dst)
        if source is None or target is None:
            continue
        evaluation = evaluator.validate_edge(source.text, target.text)
        evaluation.src = relation.src
        evaluation.dst = relation.dst
        relation.metadata["confidence"] = evaluation.confidence
        relation.rationale = evaluation.reasoning
        if evaluation.evaluation in {"support", "attack"}:
            relation.kind = _normalize_relation_kind(evaluation.evaluation)
        results.append(evaluation)
    return results


def _token_overlap(left: str, right: str) -> float:
    left_tokens = _tokenize(left)
    right_tokens = _tokenize(right)
    if not left_tokens or not right_tokens:
        return 0.0
    intersection = left_tokens & right_tokens
    union = left_tokens | right_tokens
    return len(intersection) / len(union)


def _tokenize(text: str) -> set[str]:
    return {token for token in _split_words(text) if token}


def _split_words(text: str) -> Iterable[str]:
    current = []
    for char in text.lower():
        if char.isalnum():
            current.append(char)
        elif current:
            yield "".join(current)
            current = []
    if current:
        yield "".join(current)


def _evidence_text(item: EvidenceItem) -> str:
    source = item.source
    if isinstance(source, dict):
        return str(source.get("text") or source.get("excerpt") or source)
    return source.text


def _normalize_stance(value: str) -> Literal["supports", "attacks", "neutral"]:
    if value == "attacks":
        return "attacks"
    if value == "neutral":
        return "neutral"
    return "supports"


def _normalize_relation_kind(
    value: str,
) -> Literal["support", "attack", "undercut", "rebut"]:
    if value == "attack":
        return "attack"
    return "support"
