"""LLM-assisted claim credibility scoring."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from arglib.core import ArgumentGraph, ArgumentUnit, EvidenceCard, EvidenceItem

from .llm import LLMHook, PromptTemplate


@dataclass
class EvidenceScore:
    evidence_id: str
    trust: float
    score: float
    rationale: str | None = None


@dataclass
class ClaimCredibilityResult:
    unit_id: str
    claim_score: float
    weighted_score: float
    score_source: str
    evidence_scores: list[EvidenceScore]
    raw_response: str


CLAIM_CREDIBILITY_TEMPLATE = PromptTemplate(
    system=(
        "You are an expert in argument analysis. Score how evidence supports or "
        "contradicts a claim. Evidence trust values represent reliability and must "
        "be considered when judging overall claim credibility."
    ),
    user=(
        "Claim:\n{claim}\n\n"
        "Evidence items (trust in [0,1]):\n{evidence}\n\n"
        "Return JSON with this schema:\n"
        "{{\n"
        '  "claim_score": <float -1 to 1>,\n'
        '  "evidence": [\n'
        '    {{"evidence_id": "...", "score": <float -1 to 1>, "rationale": "..."}}\n'
        "  ],\n"
        '  "summary": "..." \n'
        "}}\n\n"
        "Rules:\n"
        "- Score each evidence item on how it supports or contradicts the claim.\n"
        "- Provide a concise rationale for each evidence item.\n"
        "- The overall claim_score should reflect the combined evidence and trust.\n"
        "- If evidence does not directly address the claim, score should be near 0.\n"
        "- If evidence directly contradicts the claim, score should be negative.\n"
        "- Ensure the rationale is consistent with the score (no mismatch).\n"
    ),
)


def build_claim_credibility_hook(client) -> LLMHook:
    return LLMHook(client=client, template=CLAIM_CREDIBILITY_TEMPLATE)


def score_claims_with_llm(
    graph: ArgumentGraph,
    hook: LLMHook,
    *,
    unit_ids: Iterable[str] | None = None,
    use_llm_score: bool = True,
    persist: bool = True,
) -> dict[str, ClaimCredibilityResult]:
    results: dict[str, ClaimCredibilityResult] = {}
    target_ids = list(unit_ids or graph.units.keys())

    for unit_id in target_ids:
        unit = graph.units[unit_id]
        evidence_inputs = _collect_evidence_inputs(unit, graph)
        if not evidence_inputs:
            result = ClaimCredibilityResult(
                unit_id=unit_id,
                claim_score=0.0,
                weighted_score=0.0,
                score_source="none",
                evidence_scores=[],
                raw_response="",
            )
            results[unit_id] = result
            if persist:
                _persist_claim_score(unit, result)
            continue

        evidence_lines = "\n".join(
            f"- id: {item['evidence_id']}\n"
            f"  trust: {item['trust']:.3f}\n"
            f"  excerpt: {item['excerpt']}"
            for item in evidence_inputs
        )
        prompt_context = {"claim": unit.text, "evidence": evidence_lines}
        raw = hook.run(input="claim-credibility", context=prompt_context)
        parsed = _parse_claim_credibility(raw)
        evidence_scores = _build_evidence_scores(parsed, evidence_inputs)
        weighted_score = _weighted_score(evidence_scores)
        claim_score, score_source = _select_claim_score(
            parsed, weighted_score, use_llm_score
        )

        result = ClaimCredibilityResult(
            unit_id=unit_id,
            claim_score=claim_score,
            weighted_score=weighted_score,
            score_source=score_source,
            evidence_scores=evidence_scores,
            raw_response=raw,
        )
        results[unit_id] = result
        if persist:
            _persist_claim_score(unit, result)

    return results


def _collect_evidence_inputs(
    unit: ArgumentUnit, graph: ArgumentGraph
) -> list[dict[str, Any]]:
    inputs: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for item in unit.evidence:
        if isinstance(item, EvidenceItem):
            evidence_id = item.id
            if evidence_id in seen_ids:
                continue
            card = graph.evidence_cards.get(evidence_id)
            if card is not None:
                inputs.append(_card_to_input(card))
            else:
                inputs.append(
                    {
                        "evidence_id": evidence_id,
                        "excerpt": _evidence_text(item),
                        "trust": 1.0,
                    }
                )
            seen_ids.add(evidence_id)

    for evidence_id in unit.evidence_ids:
        if evidence_id in seen_ids:
            continue
        card = graph.evidence_cards.get(evidence_id)
        if card is None:
            continue
        inputs.append(_card_to_input(card))
        seen_ids.add(evidence_id)

    return inputs


def _card_to_input(card: EvidenceCard) -> dict[str, Any]:
    trust = max(0.0, min(1.0, float(card.confidence)))
    return {
        "evidence_id": card.id,
        "excerpt": card.excerpt,
        "trust": trust,
    }


def _parse_claim_credibility(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and start < end:
        snippet = text[start : end + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            return {}
    return {}


def _build_evidence_scores(
    parsed: dict[str, Any], evidence_inputs: list[dict[str, Any]]
) -> list[EvidenceScore]:
    scores_by_id = {}
    for item in parsed.get("evidence", []) or []:
        evidence_id = str(item.get("evidence_id", "")).strip()
        if not evidence_id:
            continue
        score = _clamp(float(item.get("score", 0.0)), -1.0, 1.0)
        scores_by_id[evidence_id] = EvidenceScore(
            evidence_id=evidence_id,
            trust=0.0,
            score=score,
            rationale=str(item.get("rationale", "")).strip() or None,
        )

    results: list[EvidenceScore] = []
    for entry in evidence_inputs:
        evidence_id = entry["evidence_id"]
        trust = _clamp(float(entry.get("trust", 1.0)), 0.0, 1.0)
        base = scores_by_id.get(evidence_id)
        score = base.score if base else 0.0
        rationale = base.rationale if base else None
        score = _apply_rationale_consistency(score, rationale)
        results.append(
            EvidenceScore(
                evidence_id=evidence_id,
                trust=trust,
                score=score,
                rationale=rationale,
            )
        )
    return results


def _weighted_score(evidence_scores: list[EvidenceScore]) -> float:
    total_weight = sum(score.trust for score in evidence_scores)
    if total_weight <= 0:
        return 0.0
    weighted = (
        sum(score.trust * score.score for score in evidence_scores) / total_weight
    )
    return _clamp(weighted, -1.0, 1.0)


def _select_claim_score(
    parsed: dict[str, Any], weighted_score: float, use_llm_score: bool
) -> tuple[float, str]:
    if use_llm_score and "claim_score" in parsed:
        try:
            claim_score = _clamp(float(parsed["claim_score"]), -1.0, 1.0)
        except (TypeError, ValueError):
            return weighted_score, "weighted"
        if abs(claim_score) > 0.2 and abs(weighted_score) < 0.2:
            return weighted_score, "weighted_consistency"
        return claim_score, "llm"
    return weighted_score, "weighted"


def _persist_claim_score(unit: ArgumentUnit, result: ClaimCredibilityResult) -> None:
    unit.metadata["claim_credibility"] = result.claim_score
    unit.metadata["claim_credibility_weighted"] = result.weighted_score
    unit.metadata["claim_credibility_method"] = result.score_source
    unit.metadata["claim_credibility_evidence"] = [
        {
            "evidence_id": score.evidence_id,
            "trust": score.trust,
            "score": score.score,
            "rationale": score.rationale,
        }
        for score in result.evidence_scores
    ]


def _evidence_text(item: EvidenceItem) -> str:
    source = item.source
    if isinstance(source, dict):
        for key in ("excerpt", "text", "content"):
            if key in source and isinstance(source[key], str):
                return source[key]
        return json.dumps(source, ensure_ascii=True)
    return str(source)


def _apply_rationale_consistency(score: float, rationale: str | None) -> float:
    if not rationale:
        return score
    text = rationale.lower()
    mismatch_terms = (
        "does not directly",
        "doesn't directly",
        "not directly",
        "indirect",
        "tangential",
        "unrelated",
        "not relevant",
        "irrelevant",
    )
    if any(term in text for term in mismatch_terms):
        return 0.0
    return score


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
