"""LLM-assisted claim type classification."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .llm import LLMHook, PromptTemplate


@dataclass
class ClaimTypeResult:
    claim_type: str
    confidence: float | None = None
    rationale: str | None = None


CLAIM_TYPE_TEMPLATE = PromptTemplate(
    system=(
        "You are an expert in argument analysis. Classify claim type as "
        "factual, value, policy, or other."
    ),
    user=(
        "Claim:\n{claim}\n\n"
        "Return JSON with this schema:\n"
        "{{\n"
        '  "claim_type": "factual|value|policy|other",\n'
        '  "confidence": <float 0 to 1>,\n'
        '  "rationale": "..." \n'
        "}}\n\n"
        "Rules:\n"
        "- factual: descriptive or empirical statements.\n"
        "- value: evaluative or normative statements.\n"
        "- policy: recommends action or rule.\n"
        "- other: does not fit above categories.\n"
    ),
)


def build_claim_type_hook(client) -> LLMHook:
    return LLMHook(client=client, template=CLAIM_TYPE_TEMPLATE)


def classify_claim_type(
    *,
    claim: str,
    hook: LLMHook,
) -> ClaimTypeResult:
    raw = hook.run(input="claim-type", context={"claim": claim})
    parsed = _parse_claim_type(raw)
    claim_type = _normalize_type(str(parsed.get("claim_type", "other")))
    confidence = parsed.get("confidence")
    confidence_value = None
    if isinstance(confidence, (int, float)):
        confidence_value = _clamp(float(confidence), 0.0, 1.0)
    rationale = str(parsed.get("rationale", "")).strip() or None
    return ClaimTypeResult(
        claim_type=claim_type,
        confidence=confidence_value,
        rationale=rationale,
    )


def _parse_claim_type(raw: str) -> dict[str, Any]:
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


def _normalize_type(value: str) -> str:
    lowered = value.lower()
    if lowered in {"factual", "fact"}:
        return "fact"
    if lowered in {"value"}:
        return "value"
    if lowered in {"policy"}:
        return "policy"
    return "other"


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


__all__ = [
    "CLAIM_TYPE_TEMPLATE",
    "ClaimTypeResult",
    "build_claim_type_hook",
    "classify_claim_type",
]
