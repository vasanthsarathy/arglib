"""Warrant-gated reasoning over claim graphs."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp

from arglib.core import ArgumentGraph, ArgumentUnit, EvidenceCard, EvidenceItem, Warrant


@dataclass
class WarrantGatedConfig:
    alpha: float = 1.0
    beta: float = 1.0
    max_iterations: int = 50
    epsilon: float = 1e-4
    gate_required: bool = True
    clamp_influence: float = 4.0


@dataclass
class WarrantGatedResult:
    evidence_support: dict[str, float]
    warrant_evidence_support: dict[str, float]
    gate_scores: dict[str, float]
    claim_iterations: list[dict[str, float]]
    warrant_iterations: list[dict[str, float]]
    final_claim_scores: dict[str, float]
    final_warrant_scores: dict[str, float]


def compute_warrant_gated_scores(
    graph: ArgumentGraph, *, config: WarrantGatedConfig | None = None
) -> WarrantGatedResult:
    cfg = config or WarrantGatedConfig()
    claim_ev = {
        unit_id: _evidence_support(unit, graph)
        for unit_id, unit in graph.units.items()
    }
    warrant_ev = {
        warrant_id: _evidence_support(warrant, graph)
        for warrant_id, warrant in graph.warrants.items()
    }
    for unit_id, unit in graph.units.items():
        if unit.is_axiom:
            base = _clamp(
                float(unit.score) if unit.score is not None else 0.0, -1.0, 1.0
            )
            claim_ev[unit_id] = base
    for warrant_id, warrant in graph.warrants.items():
        if warrant.is_axiom:
            base = _clamp(
                float(warrant.score) if warrant.score is not None else 0.0, -1.0, 1.0
            )
            warrant_ev[warrant_id] = base

    claim_scores = {
        unit_id: _sigmoid(cfg.alpha * ev) for unit_id, ev in claim_ev.items()
    }
    warrant_scores = {
        warrant_id: _sigmoid(cfg.alpha * ev) for warrant_id, ev in warrant_ev.items()
    }

    claim_iterations = [dict(claim_scores)]
    warrant_iterations = [dict(warrant_scores)]

    for _ in range(cfg.max_iterations):
        new_warrant_scores = _update_warrants(graph, warrant_ev, claim_scores, cfg)
        gate_scores = _gate_scores(graph, new_warrant_scores, cfg)
        new_claim_scores = _update_claims(
            graph, claim_ev, claim_scores, gate_scores, cfg
        )

        claim_iterations.append(dict(new_claim_scores))
        warrant_iterations.append(dict(new_warrant_scores))

        max_change = 0.0
        for key, value in new_claim_scores.items():
            max_change = max(max_change, abs(value - claim_scores.get(key, 0.0)))
        for key, value in new_warrant_scores.items():
            max_change = max(max_change, abs(value - warrant_scores.get(key, 0.0)))

        claim_scores = new_claim_scores
        warrant_scores = new_warrant_scores
        if max_change < cfg.epsilon:
            break

    gate_scores = _gate_scores(graph, warrant_scores, cfg)
    return WarrantGatedResult(
        evidence_support=claim_ev,
        warrant_evidence_support=warrant_ev,
        gate_scores=gate_scores,
        claim_iterations=claim_iterations,
        warrant_iterations=warrant_iterations,
        final_claim_scores=claim_scores,
        final_warrant_scores=warrant_scores,
    )


def _evidence_support(node: ArgumentUnit | Warrant, graph: ArgumentGraph) -> float:
    signed_scores: list[float] = []
    seen_ids: set[str] = set()

    for item in getattr(node, "evidence", []):
        if not isinstance(item, EvidenceItem):
            continue
        evidence_id = item.id
        score = _evidence_item_score(item, graph)
        if evidence_id in seen_ids:
            continue
        signed_scores.append(score)
        seen_ids.add(evidence_id)

    for evidence_id in getattr(node, "evidence_ids", []):
        if evidence_id in seen_ids:
            continue
        card = graph.evidence_cards.get(evidence_id)
        if card is None:
            continue
        signed_scores.append(_card_score(card, graph, stance="supports"))
        seen_ids.add(evidence_id)

    if not signed_scores:
        return 0.0
    value = sum(signed_scores) / len(signed_scores)
    return _clamp(value, -1.0, 1.0)


def _evidence_item_score(item: EvidenceItem, graph: ArgumentGraph) -> float:
    stance = item.stance
    sign = 1.0 if stance == "supports" else -1.0 if stance == "attacks" else 0.0
    if sign == 0.0:
        return 0.0
    evidence_id = item.id
    card = graph.evidence_cards.get(evidence_id)
    if card is not None:
        return _card_score(card, graph, stance=stance)
    strength = item.strength if item.strength is not None else 1.0
    return sign * _clamp(float(strength), 0.0, 1.0)


def _card_score(
    card: EvidenceCard, graph: ArgumentGraph, *, stance: str
) -> float:
    sign = 1.0 if stance == "supports" else -1.0 if stance == "attacks" else 0.0
    trust = 1.0
    doc_id = getattr(card, "supporting_doc_id", None)
    if doc_id and doc_id in graph.supporting_documents:
        doc = graph.supporting_documents[doc_id]
        if doc.trust is not None:
            trust = _clamp(float(doc.trust), 0.0, 1.0)
    confidence = _clamp(float(card.confidence), 0.0, 1.0)
    return sign * trust * confidence


def _update_warrants(
    graph: ArgumentGraph,
    warrant_ev: dict[str, float],
    claim_scores: dict[str, float],
    cfg: WarrantGatedConfig,
) -> dict[str, float]:
    incoming: dict[str, list[float]] = {w: [] for w in graph.warrants}
    ignore_warrants = {
        warrant_id
        for warrant_id, warrant in graph.warrants.items()
        if warrant.ignore_influence
    }
    for attack in graph.warrant_attacks:
        if attack.warrant_id in ignore_warrants:
            continue
        src_score = claim_scores.get(attack.src, 0.0)
        incoming.setdefault(attack.warrant_id, []).append(-src_score)

    new_scores: dict[str, float] = {}
    for warrant_id in graph.warrants:
        ev = warrant_ev.get(warrant_id, 0.0)
        if warrant_id in ignore_warrants:
            influence = 0.0
        else:
            influence = sum(incoming.get(warrant_id, []))
        z = cfg.alpha * ev + cfg.beta * _clamp(
            influence, -cfg.clamp_influence, cfg.clamp_influence
        )
        new_scores[warrant_id] = _sigmoid(z)
    return new_scores


def _gate_scores(
    graph: ArgumentGraph,
    warrant_scores: dict[str, float],
    cfg: WarrantGatedConfig,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for index, relation in enumerate(graph.relations):
        key = f"e{index}"
        if relation.metadata.get("gate_disabled"):
            scores[key] = 0.0
            continue
        if not relation.warrant_ids:
            scores[key] = 0.0 if cfg.gate_required else 1.0
            continue
        values = [warrant_scores.get(wid, 0.5) for wid in relation.warrant_ids]
        if relation.gate_mode == "AND":
            scores[key] = min(values)
        else:
            scores[key] = max(values)
    return scores


def _update_claims(
    graph: ArgumentGraph,
    claim_ev: dict[str, float],
    claim_scores: dict[str, float],
    gate_scores: dict[str, float],
    cfg: WarrantGatedConfig,
) -> dict[str, float]:
    incoming: dict[str, list[float]] = {unit_id: [] for unit_id in graph.units}
    ignore_units = {
        unit_id
        for unit_id, unit in graph.units.items()
        if unit.ignore_influence
    }
    for index, relation in enumerate(graph.relations):
        if relation.dst in ignore_units:
            continue
        sign = 1.0 if relation.kind == "support" else -1.0
        src_score = claim_scores.get(relation.src, 0.0)
        gate = gate_scores.get(f"e{index}", 0.0)
        influence = sign * src_score * gate
        incoming.setdefault(relation.dst, []).append(influence)

    new_scores: dict[str, float] = {}
    for unit_id in graph.units:
        ev = claim_ev.get(unit_id, 0.0)
        if unit_id in ignore_units:
            influence = 0.0
        else:
            influence = sum(incoming.get(unit_id, []))
        z = cfg.alpha * ev + cfg.beta * _clamp(
            influence, -cfg.clamp_influence, cfg.clamp_influence
        )
        new_scores[unit_id] = _sigmoid(z)
    return new_scores


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + exp(-value))


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


__all__ = [
    "WarrantGatedConfig",
    "WarrantGatedResult",
    "compute_warrant_gated_scores",
]
