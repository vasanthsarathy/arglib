"""Explain warrant-gated credibility scores."""

from __future__ import annotations

from typing import Any

from arglib.core import ArgumentGraph
from arglib.reasoning.credibility import CredibilityResult


def explain_credibility(
    graph: ArgumentGraph,
    result: CredibilityResult,
) -> dict[str, Any]:
    explanations: dict[str, Any] = {}
    for claim_id in graph.units:
        explanations[claim_id] = {
            "evidence_support": result.initial_evidence.get(claim_id, 0.0),
            "incoming": [],
            "total_influence": 0.0,
            "final_score": result.final_scores.get(claim_id, 0.0),
        }

    for index, relation in enumerate(graph.relations):
        edge_id = f"e{index}"
        gate = result.gate_scores.get(edge_id, 0.0)
        src_score = result.final_scores.get(relation.src, 0.0)
        sign = 1.0 if relation.kind == "support" else -1.0
        influence = sign * src_score * gate
        entry = {
            "edge_id": edge_id,
            "src": relation.src,
            "dst": relation.dst,
            "kind": relation.kind,
            "gate_score": gate,
            "src_score": src_score,
            "influence": influence,
            "warrant_ids": list(relation.warrant_ids),
            "gate_mode": relation.gate_mode,
        }
        if relation.dst in explanations:
            explanations[relation.dst]["incoming"].append(entry)
            explanations[relation.dst]["total_influence"] += influence

    return explanations
