"""LLM-assisted edge validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .llm import LLMHook, PromptTemplate


@dataclass
class EdgeValidationLLMResult:
    evaluation: str
    score: float
    rationale: str | None = None


EDGE_VALIDATION_TEMPLATE = PromptTemplate(
    system=(
        "You are an expert in argument analysis. Evaluate the logical relation "
        "between two claims as support, attack, or neutral."
    ),
    user=(
        "Source claim:\n{source}\n\n"
        "Target claim:\n{target}\n\n"
        "Return JSON with this schema:\n"
        "{{\n"
        '  "evaluation": "support|attack|neutral",\n'
        '  "score": <float -1 to 1>,\n'
        '  "rationale": "..." \n'
        "}}\n\n"
        "Rules:\n"
        "- score > 0 means support, score < 0 means attack, score ~= 0 neutral.\n"
        "- rationale must match the evaluation.\n"
    ),
)


def build_edge_validation_hook(client) -> LLMHook:
    return LLMHook(client=client, template=EDGE_VALIDATION_TEMPLATE)


def validate_edge_with_llm(
    *,
    source: str,
    target: str,
    hook: LLMHook,
) -> EdgeValidationLLMResult:
    context = {"source": source, "target": target}
    raw = hook.run(input="edge-validation", context=context)
    parsed = _parse_edge_validation(raw)
    evaluation = _normalize_eval(str(parsed.get("evaluation", "neutral")))
    score = _clamp(float(parsed.get("score", 0.0)), -1.0, 1.0)
    rationale = str(parsed.get("rationale", "")).strip() or None
    return EdgeValidationLLMResult(
        evaluation=evaluation,
        score=score,
        rationale=rationale,
    )


def _parse_edge_validation(raw: str) -> dict[str, Any]:
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


def _normalize_eval(value: str) -> str:
    lowered = value.lower()
    if lowered in {"support", "supports"}:
        return "support"
    if lowered in {"attack", "attacks"}:
        return "attack"
    return "neutral"


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


__all__ = [
    "EDGE_VALIDATION_TEMPLATE",
    "EdgeValidationLLMResult",
    "build_edge_validation_hook",
    "validate_edge_with_llm",
]
