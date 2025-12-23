"""Credibility propagation for argument graphs."""

from __future__ import annotations

from dataclasses import dataclass
from math import tanh

from arglib.core import (
    ArgumentGraph,
    ArgumentUnit,
    EvidenceCard,
    EvidenceItem,
    Relation,
)


@dataclass
class CredibilityResult:
    initial_evidence: dict[str, float]
    iterations: list[dict[str, float]]
    final_scores: dict[str, float]


def compute_credibility(
    graph: ArgumentGraph,
    *,
    lambda_: float = 0.5,
    max_iterations: int = 100,
    epsilon: float = 1e-6,
    evidence_min: float = 0.0,
    evidence_max: float = 1.0,
) -> CredibilityResult:
    evidence_scores: dict[str, float] = {}
    for unit_id, unit in graph.units.items():
        scores = _collect_evidence_scores(unit, graph)
        if scores:
            min_val = (
                unit.evidence_min if unit.evidence_min is not None else evidence_min
            )
            max_val = (
                unit.evidence_max if unit.evidence_max is not None else evidence_max
            )
            clamped = [max(min(score, max_val), min_val) for score in scores]
            evidence_scores[unit_id] = sum(clamped) / len(clamped)
        else:
            evidence_scores[unit_id] = 0.0

    c_prev = dict(evidence_scores)
    iterations = [c_prev.copy()]

    incoming_edges: dict[str, list[Relation]] = {unit_id: [] for unit_id in graph.units}
    for relation in graph.relations:
        incoming_edges.setdefault(relation.dst, []).append(relation)

    source_nodes = {node_id for node_id, edges in incoming_edges.items() if not edges}

    for _ in range(max_iterations):
        c_new = c_prev.copy()
        for node_id, edges in incoming_edges.items():
            if node_id in source_nodes or not edges:
                continue
            total_edge_contribution = 0.0
            meaningful = False
            for edge in edges:
                contribution = _edge_contribution(edge, c_prev)
                total_edge_contribution += contribution
                if abs(contribution) > 0.001:
                    meaningful = True
            if meaningful:
                z = lambda_ * evidence_scores[node_id] + total_edge_contribution
                c_new[node_id] = tanh(z)
            else:
                c_new[node_id] = c_prev[node_id]

        iterations.append(c_new.copy())
        if max(abs(c_new[n] - c_prev[n]) for n in c_new) < epsilon:
            break
        c_prev = c_new

    final_scores = iterations[-1]
    return CredibilityResult(
        initial_evidence=evidence_scores,
        iterations=iterations,
        final_scores=final_scores,
    )


def _collect_evidence_scores(unit: ArgumentUnit, graph: ArgumentGraph) -> list[float]:
    scores: list[float] = []
    seen_ids: set[str] = set()

    for item in getattr(unit, "evidence", []):
        if isinstance(item, EvidenceItem):
            if item.id in graph.evidence_cards:
                scores.append(graph.evidence_cards[item.id].confidence)
                seen_ids.add(item.id)
            elif item.strength is not None:
                scores.append(item.strength)
                seen_ids.add(item.id)

    for evidence_id in getattr(unit, "evidence_ids", []):
        if evidence_id in seen_ids:
            continue
        card: EvidenceCard | None = graph.evidence_cards.get(evidence_id)
        if card is not None:
            scores.append(card.confidence)
            seen_ids.add(evidence_id)

    return scores


def _edge_contribution(edge: Relation, scores: dict[str, float]) -> float:
    if edge.weight == 0.0:
        return 0.0
    weight = 1.0 if edge.weight is None else edge.weight
    strength = abs(weight)
    source_score = scores.get(edge.src, 0.0)
    if edge.kind == "support":
        return strength * source_score
    return -strength * abs(source_score)
