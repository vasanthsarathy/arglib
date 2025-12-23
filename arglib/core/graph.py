"""Argument graph model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from .evidence import EvidenceItem
from .relations import Relation
from .spans import TextSpan
from .units import ArgumentUnit

if TYPE_CHECKING:
    from arglib.semantics import DungAF

@dataclass
class ArgumentGraph:
    units: dict[str, ArgumentUnit] = field(default_factory=dict)
    relations: list[Relation] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def new(cls, title: str | None = None) -> ArgumentGraph:
        metadata: dict[str, Any] = {}
        if title:
            metadata["title"] = title
        return cls(metadata=metadata)

    def add_claim(
        self,
        text: str,
        claim_id: str | None = None,
        type: Literal["fact", "value", "policy", "other"] = "other",
        spans: list[TextSpan] | None = None,
        evidence: list[EvidenceItem] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        if claim_id is None:
            claim_id = self._next_id("c")
        unit = ArgumentUnit(
            id=claim_id,
            text=text,
            type=type,
            spans=list(spans or []),
            evidence=list(evidence or []),
            metadata=dict(metadata or {}),
        )
        self.units[claim_id] = unit
        return claim_id

    def add_relation(
        self,
        src: str,
        dst: str,
        kind: Literal["support", "attack", "undercut", "rebut"],
        weight: float | None = None,
        rationale: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Relation:
        relation = Relation(
            src=src,
            dst=dst,
            kind=kind,
            weight=weight,
            rationale=rationale,
            metadata=dict(metadata or {}),
        )
        self.relations.append(relation)
        return relation

    def add_support(self, src: str, dst: str, **kwargs: Any) -> Relation:
        return self.add_relation(src, dst, kind="support", **kwargs)

    def add_attack(self, src: str, dst: str, **kwargs: Any) -> Relation:
        return self.add_relation(src, dst, kind="attack", **kwargs)

    def attach_evidence(
        self,
        unit_id: str,
        evidence_id: str,
        source: TextSpan | dict[str, Any],
        stance: Literal["supports", "attacks", "neutral"],
        strength: float | None = None,
        quality: dict[str, Any] | None = None,
    ) -> EvidenceItem:
        if unit_id not in self.units:
            raise KeyError(f"Unknown unit id: {unit_id}")
        item = EvidenceItem(
            id=evidence_id,
            source=source,
            stance=stance,
            strength=strength,
            quality=dict(quality or {}),
        )
        self.units[unit_id].evidence.append(item)
        return item

    def to_dict(self) -> dict[str, Any]:
        return {
            "units": {unit_id: unit.to_dict() for unit_id, unit in self.units.items()},
            "relations": [relation.to_dict() for relation in self.relations],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArgumentGraph:
        units = {
            unit_id: ArgumentUnit.from_dict(unit_data)
            for unit_id, unit_data in data.get("units", {}).items()
        }
        relations = [Relation.from_dict(item) for item in data.get("relations", [])]
        metadata = data.get("metadata", {})
        return cls(units=units, relations=relations, metadata=metadata)

    def to_dung(self, include_relations: list[str] | None = None) -> DungAF:
        from arglib.semantics import DungAF

        if include_relations is None:
            include_relations = ["attack", "undercut", "rebut"]

        af = DungAF(arguments=set(self.units.keys()))
        for relation in self.relations:
            if relation.kind in include_relations:
                af.add_attack(relation.src, relation.dst)
        return af

    def diagnostics(self) -> dict[str, Any]:
        from arglib.algorithms import (
            build_edges,
            find_cycles,
            in_out_degree,
            reachability_map,
            strongly_connected_components,
            weakly_connected_components,
        )

        nodes = list(self.units.keys())
        relations = [(rel.src, rel.dst, rel.kind) for rel in self.relations]
        all_edges = build_edges(relations)
        cycles = find_cycles(nodes, all_edges)
        components = weakly_connected_components(nodes, all_edges)
        sccs = strongly_connected_components(nodes, all_edges)
        degrees = in_out_degree(nodes, all_edges)
        reachability = reachability_map(nodes, all_edges)

        isolated = [
            node
            for node, degree in degrees.items()
            if degree["in"] == 0 and degree["out"] == 0
        ]
        support_edges = build_edges(relations, kinds=["support"])
        support_degrees = in_out_degree(nodes, support_edges)
        unsupported = [
            node for node, degree in support_degrees.items() if degree["in"] == 0
        ]
        attack_edges = build_edges(relations, kinds=["attack", "undercut", "rebut"])

        node_count = len(nodes)
        relation_count = len(self.relations)
        degree_totals = [degree["in"] for degree in degrees.values()]
        out_totals = [degree["out"] for degree in degrees.values()]
        avg_in = sum(degree_totals) / node_count if node_count else 0.0
        avg_out = sum(out_totals) / node_count if node_count else 0.0

        return {
            "node_count": node_count,
            "relation_count": relation_count,
            "attack_edge_count": len(attack_edges),
            "support_edge_count": len(support_edges),
            "cycle_count": len(cycles),
            "cycles": cycles,
            "component_count": len(components),
            "components": components,
            "scc_count": len(sccs),
            "strongly_connected_components": sccs,
            "isolated_units": sorted(isolated),
            "unsupported_claims": sorted(unsupported),
            "degrees": degrees,
            "degree_summary": {
                "avg_in": avg_in,
                "avg_out": avg_out,
                "max_in": max(degree_totals, default=0),
                "max_out": max(out_totals, default=0),
            },
            "reachability": {
                node: sorted(targets) for node, targets in reachability.items()
            },
            "max_reachability": max(
                (len(targets) for targets in reachability.values()), default=0
            ),
        }

    def critique(self) -> dict[str, Any]:
        from arglib.critique import suggest_missing_assumptions

        return {
            "missing_assumptions": suggest_missing_assumptions(self),
        }

    def _next_id(self, prefix: str) -> str:
        max_id = 0
        for unit_id in self.units:
            if unit_id.startswith(prefix):
                suffix = unit_id[len(prefix) :]
                if suffix.isdigit():
                    max_id = max(max_id, int(suffix))
        return f"{prefix}{max_id + 1}"
