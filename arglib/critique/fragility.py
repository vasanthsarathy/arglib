"""Warrant fragility analysis for warrant-gated graphs."""

from __future__ import annotations

from dataclasses import dataclass

from arglib.core import ArgumentGraph
from arglib.reasoning import compute_credibility


@dataclass(frozen=True)
class WarrantFragility:
    edge_id: str
    dst: str
    gate_mode: str
    gate_score: float
    critical_warrants: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "edge_id": self.edge_id,
            "dst": self.dst,
            "gate_mode": self.gate_mode,
            "gate_score": self.gate_score,
            "critical_warrants": list(self.critical_warrants),
        }


def analyze_warrant_fragility(
    graph: ArgumentGraph,
) -> list[WarrantFragility]:
    result = compute_credibility(graph)
    warrant_scores = result.warrant_scores
    gate_scores = result.gate_scores

    fragility: list[WarrantFragility] = []
    for index, relation in enumerate(graph.relations):
        if not relation.warrant_ids:
            continue
        scores = [warrant_scores.get(wid, 0.5) for wid in relation.warrant_ids]
        if not scores:
            continue
        edge_id = f"e{index}"
        if relation.gate_mode == "AND":
            min_score = min(scores)
            critical = [
                wid
                for wid in relation.warrant_ids
                if warrant_scores.get(wid, 0.5) == min_score
            ]
        else:
            max_score = max(scores)
            critical = [
                wid
                for wid in relation.warrant_ids
                if warrant_scores.get(wid, 0.5) == max_score
            ]
        fragility.append(
            WarrantFragility(
                edge_id=edge_id,
                dst=relation.dst,
                gate_mode=relation.gate_mode,
                gate_score=gate_scores.get(edge_id, 0.0),
                critical_warrants=critical,
            )
        )

    return fragility
