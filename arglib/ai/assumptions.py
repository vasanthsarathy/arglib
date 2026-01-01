"""LLM-assisted implicit assumption generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .llm import LLMHook, PromptTemplate


@dataclass
class AssumptionResult:
    assumption: str
    rationale: str | None = None
    importance: float | None = None


ASSUMPTION_TEMPLATE = PromptTemplate(
    system=(
        "You are an expert in argument analysis. Generate implicit, necessary "
        "assumptions that must hold for a relation between two claims to be valid."
    ),
    user=(
        "Source claim:\n{source}\n\n"
        "Target claim:\n{target}\n\n"
        "Relation type: {relation}\n"
        "Generate exactly {k} implicit assumptions that are necessary for this "
        "relation to hold. Return JSON with this schema:\n"
        "{{\n"
        '  "assumptions": [\n'
        '    {{"assumption": "...", "rationale": "...", "importance": 0.0}}\n'
        "  ]\n"
        "}}\n\n"
        "Rules:\n"
        "- Each assumption should be a concrete, testable statement.\n"
        "- Keep assumptions necessary (not merely helpful).\n"
        "- importance is optional; if present it must be between 0 and 1.\n"
    ),
)


def build_assumption_hook(client) -> LLMHook:
    return LLMHook(client=client, template=ASSUMPTION_TEMPLATE)


def generate_edge_assumptions(
    *,
    source: str,
    target: str,
    relation: str,
    k: int,
    hook: LLMHook,
) -> list[AssumptionResult]:
    context = {"source": source, "target": target, "relation": relation, "k": k}
    raw = hook.run(input="edge-assumptions", context=context)
    parsed = _parse_assumptions(raw)
    return _normalize_assumptions(parsed, k)


def _parse_assumptions(raw: str) -> list[dict[str, Any]]:
    text = raw.strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None

    if payload is None:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and start < end:
            try:
                payload = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                payload = None

    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        items = payload.get("assumptions")
        if isinstance(items, list):
            return items
    return []


def _normalize_assumptions(
    items: list[dict[str, Any]], k: int
) -> list[AssumptionResult]:
    results: list[AssumptionResult] = []
    for item in items:
        assumption = str(item.get("assumption", "")).strip()
        if not assumption:
            continue
        rationale_raw = item.get("rationale")
        rationale = str(rationale_raw).strip() if rationale_raw else None
        importance = item.get("importance")
        importance_value = None
        if isinstance(importance, (int, float)):
            importance_value = min(max(float(importance), 0.0), 1.0)
        results.append(
            AssumptionResult(
                assumption=assumption,
                rationale=rationale,
                importance=importance_value,
            )
        )
        if len(results) >= k:
            break
    return results


__all__ = [
    "ASSUMPTION_TEMPLATE",
    "AssumptionResult",
    "build_assumption_hook",
    "generate_edge_assumptions",
]
