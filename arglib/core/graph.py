"""Argument graph model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .bundles import ArgumentBundle, ArgumentBundleGraph
from .evidence import EvidenceCard, EvidenceItem, SupportingDocument
from .relations import Relation
from .spans import TextSpan
from .units import ArgumentUnit
from .warrants import Warrant, WarrantAttack


@dataclass
class ArgumentGraph:
    units: dict[str, ArgumentUnit] = field(default_factory=dict)
    warrants: dict[str, Warrant] = field(default_factory=dict)
    relations: list[Relation] = field(default_factory=list)
    warrant_attacks: list[WarrantAttack] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    evidence_cards: dict[str, EvidenceCard] = field(default_factory=dict)
    supporting_documents: dict[str, SupportingDocument] = field(default_factory=dict)
    argument_bundles: dict[str, ArgumentBundle] = field(default_factory=dict)

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
        evidence_ids: list[str] | None = None,
        evidence_min: float | None = None,
        evidence_max: float | None = None,
        score: float | None = None,
        is_axiom: bool = False,
        ignore_influence: bool = False,
    ) -> str:
        if claim_id is None:
            claim_id = self._next_id("c")
        unit = ArgumentUnit(
            id=claim_id,
            text=text,
            type=type,
            spans=list(spans or []),
            evidence=list(evidence or []),
            evidence_ids=list(evidence_ids or []),
            evidence_min=evidence_min,
            evidence_max=evidence_max,
            score=score,
            is_axiom=is_axiom,
            ignore_influence=ignore_influence,
            metadata=dict(metadata or {}),
        )
        self.units[claim_id] = unit
        return claim_id

    def add_relation(
        self,
        src: str,
        dst: str,
        kind: Literal["support", "attack", "undercut", "rebut"],
        warrant_ids: list[str] | None = None,
        gate_mode: Literal["AND", "OR"] = "OR",
        weight: float | None = None,
        rationale: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Relation:
        relation = Relation(
            src=src,
            dst=dst,
            kind=kind,
            warrant_ids=list(warrant_ids or []),
            gate_mode=gate_mode,
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

    def add_warrant(
        self,
        text: str,
        warrant_id: str | None = None,
        type: Literal["fact", "value", "policy", "other"] = "other",
        spans: list[TextSpan] | None = None,
        evidence: list[EvidenceItem] | None = None,
        metadata: dict[str, Any] | None = None,
        evidence_ids: list[str] | None = None,
        score: float | None = None,
        is_axiom: bool = False,
        ignore_influence: bool = False,
    ) -> str:
        if warrant_id is None:
            warrant_id = self._next_id("w")
        warrant = Warrant(
            id=warrant_id,
            text=text,
            type=type,
            spans=list(spans or []),
            evidence=list(evidence or []),
            evidence_ids=list(evidence_ids or []),
            score=score,
            is_axiom=is_axiom,
            ignore_influence=ignore_influence,
            metadata=dict(metadata or {}),
        )
        self.warrants[warrant_id] = warrant
        return warrant_id

    def attach_evidence_to_warrant(
        self,
        warrant_id: str,
        evidence_id: str,
        source: TextSpan | dict[str, Any],
        stance: Literal["supports", "attacks", "neutral"],
        strength: float | None = None,
        quality: dict[str, Any] | None = None,
    ) -> EvidenceItem:
        if warrant_id not in self.warrants:
            raise KeyError(f"Unknown warrant id: {warrant_id}")
        item = EvidenceItem(
            id=evidence_id,
            source=source,
            stance=stance,
            strength=strength,
            quality=dict(quality or {}),
        )
        self.warrants[warrant_id].evidence.append(item)
        return item

    def add_warrant_attack(
        self,
        src: str,
        warrant_id: str,
        rationale: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> WarrantAttack:
        if warrant_id not in self.warrants:
            raise KeyError(f"Unknown warrant id: {warrant_id}")
        attack = WarrantAttack(
            src=src,
            warrant_id=warrant_id,
            rationale=rationale,
            metadata=dict(metadata or {}),
        )
        self.warrant_attacks.append(attack)
        return attack

    def add_supporting_document(
        self, document: SupportingDocument, *, overwrite: bool = False
    ) -> None:
        if document.id in self.supporting_documents and not overwrite:
            raise ValueError(f"Supporting document already exists: {document.id}")
        self.supporting_documents[document.id] = document

    def add_evidence_card(
        self, evidence: EvidenceCard, *, overwrite: bool = False
    ) -> None:
        if evidence.id in self.evidence_cards and not overwrite:
            raise ValueError(f"Evidence card already exists: {evidence.id}")
        self.evidence_cards[evidence.id] = evidence

    def attach_evidence_card(
        self,
        unit_id: str,
        evidence_id: str,
        stance: Literal["supports", "attacks", "neutral"] = "supports",
    ) -> EvidenceItem:
        if unit_id not in self.units:
            raise KeyError(f"Unknown unit id: {unit_id}")
        if evidence_id not in self.evidence_cards:
            raise KeyError(f"Unknown evidence card id: {evidence_id}")
        card = self.evidence_cards[evidence_id]
        item = EvidenceItem(
            id=evidence_id,
            source={"evidence_card_id": evidence_id},
            stance=stance,
            strength=card.confidence,
            quality={},
        )
        self.units[unit_id].evidence.append(item)
        if evidence_id not in self.units[unit_id].evidence_ids:
            self.units[unit_id].evidence_ids.append(evidence_id)
        return item

    def to_dict(self) -> dict[str, Any]:
        return {
            "units": {unit_id: unit.to_dict() for unit_id, unit in self.units.items()},
            "warrants": {
                warrant_id: warrant.to_dict()
                for warrant_id, warrant in self.warrants.items()
            },
            "relations": [relation.to_dict() for relation in self.relations],
            "warrant_attacks": [attack.to_dict() for attack in self.warrant_attacks],
            "metadata": self.metadata,
            "evidence_cards": {
                evidence_id: card.to_dict()
                for evidence_id, card in self.evidence_cards.items()
            },
            "supporting_documents": {
                doc_id: doc.to_dict()
                for doc_id, doc in self.supporting_documents.items()
            },
            "argument_bundles": {
                bundle_id: bundle.to_dict()
                for bundle_id, bundle in self.argument_bundles.items()
            },
        }

    def save(self, path: str, *, indent: int = 2) -> None:
        from arglib.io import save

        save(path, self, indent=indent)

    def render(self, path: str, *, engine: str = "graphviz") -> None:
        if engine != "graphviz":
            raise ValueError(f"Unsupported engine: {engine}")
        from arglib.viz import to_dot

        with open(path, "w", encoding="utf-8") as handle:
            handle.write(to_dot(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArgumentGraph:
        units = {
            unit_id: ArgumentUnit.from_dict(unit_data)
            for unit_id, unit_data in data.get("units", {}).items()
        }
        warrants = {
            warrant_id: Warrant.from_dict(warrant_data)
            for warrant_id, warrant_data in data.get("warrants", {}).items()
        }
        relations = [Relation.from_dict(item) for item in data.get("relations", [])]
        warrant_attacks = [
            WarrantAttack.from_dict(item)
            for item in data.get("warrant_attacks", [])
        ]
        metadata = data.get("metadata", {})
        evidence_cards = {
            evidence_id: EvidenceCard.from_dict(card)
            for evidence_id, card in data.get("evidence_cards", {}).items()
        }
        supporting_documents = {
            doc_id: SupportingDocument.from_dict(doc)
            for doc_id, doc in data.get("supporting_documents", {}).items()
        }
        argument_bundles = {
            bundle_id: ArgumentBundle.from_dict(bundle)
            for bundle_id, bundle in data.get("argument_bundles", {}).items()
        }
        return cls(
            units=units,
            warrants=warrants,
            relations=relations,
            warrant_attacks=warrant_attacks,
            metadata=metadata,
            evidence_cards=evidence_cards,
            supporting_documents=supporting_documents,
            argument_bundles=argument_bundles,
        )

    def define_argument(
        self,
        units: list[str],
        relations: list[Relation] | None = None,
        bundle_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ArgumentBundle:
        if len(units) < 2:
            raise ValueError("Argument bundles require at least two units.")
        if bundle_id is None:
            bundle_id = f"arg_{len(self.argument_bundles) + 1}"
        if bundle_id in self.argument_bundles:
            raise ValueError(f"Argument bundle already exists: {bundle_id}")
        for unit_id in units:
            if unit_id not in self.units:
                raise KeyError(f"Unknown unit id: {unit_id}")
        if relations is None:
            relations = [
                rel for rel in self.relations if rel.src in units and rel.dst in units
            ]
        bundle = ArgumentBundle(
            id=bundle_id,
            units=list(units),
            relations=list(relations),
            metadata=dict(metadata or {}),
        )
        self.argument_bundles[bundle_id] = bundle
        return bundle

    def to_argument_graph(
        self,
        *,
        aggregation: str = "sum",
        clamp: bool = True,
    ) -> ArgumentBundleGraph:
        bundle_lookup: dict[str, str] = {}
        for bundle_id, bundle in self.argument_bundles.items():
            for unit_id in bundle.units:
                if unit_id in bundle_lookup:
                    raise ValueError(f"Unit {unit_id} is assigned to multiple bundles.")
                bundle_lookup[unit_id] = bundle_id

        if not bundle_lookup:
            raise ValueError("No argument bundles defined.")

        edge_scores: dict[tuple[str, str], list[float]] = {}
        for relation in self.relations:
            src_bundle = bundle_lookup.get(relation.src)
            dst_bundle = bundle_lookup.get(relation.dst)
            if not src_bundle or not dst_bundle or src_bundle == dst_bundle:
                continue
            signed = _signed_weight(relation)
            edge_scores.setdefault((src_bundle, dst_bundle), []).append(signed)

        bundle_relations: list[Relation] = []
        for (src_bundle, dst_bundle), weights in edge_scores.items():
            aggregated = _aggregate_weights(weights, aggregation)
            if clamp:
                aggregated = max(-1.0, min(1.0, aggregated))
            kind: Literal["support", "attack"] = (
                "support" if aggregated >= 0 else "attack"
            )
            bundle_relations.append(
                Relation(src=src_bundle, dst=dst_bundle, kind=kind, weight=aggregated)
            )

        return ArgumentBundleGraph(
            bundles=self.argument_bundles,
            relations=bundle_relations,
            metadata={"aggregation": aggregation, "clamp": clamp},
        )

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
        axiom_claims = sorted(
            unit_id for unit_id, unit in self.units.items() if unit.is_axiom
        )
        axiom_warrants = sorted(
            warrant_id
            for warrant_id, warrant in self.warrants.items()
            if warrant.is_axiom
        )
        axiom_warnings = [
            f"axiom claim '{unit_id}' bypasses evidence requirements."
            for unit_id in axiom_claims
        ] + [
            f"axiom warrant '{warrant_id}' bypasses evidence requirements."
            for warrant_id in axiom_warrants
        ]

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
            "axioms": {"claims": axiom_claims, "warrants": axiom_warrants},
            "axiom_warnings": axiom_warnings,
        }

    def critique(self) -> dict[str, Any]:
        from arglib.critique import (
            analyze_warrant_fragility,
            detect_patterns,
            suggest_missing_assumptions,
        )

        return {
            "missing_assumptions": suggest_missing_assumptions(self),
            "patterns": [match.to_dict() for match in detect_patterns(self)],
            "warrant_fragility": [
                item.to_dict() for item in analyze_warrant_fragility(self)
            ],
        }

    def _next_id(self, prefix: str) -> str:
        max_id = 0
        for unit_id in self.units:
            if unit_id.startswith(prefix):
                suffix = unit_id[len(prefix) :]
                if suffix.isdigit():
                    max_id = max(max_id, int(suffix))
        return f"{prefix}{max_id + 1}"


def _signed_weight(relation: Relation) -> float:
    if relation.weight is None:
        base = 1.0
    else:
        base = relation.weight
    if relation.kind == "support":
        return base
    return -abs(base)


def _aggregate_weights(weights: list[float], aggregation: str) -> float:
    if not weights:
        return 0.0
    aggregation = aggregation.lower()
    if aggregation == "sum":
        return sum(weights)
    if aggregation == "mean":
        return sum(weights) / len(weights)
    if aggregation == "max":
        return max(weights, key=abs)
    if aggregation == "softmax":
        exp_weights = [pow(2.718281828, abs(w)) for w in weights]
        total = sum(exp_weights)
        if total == 0:
            return 0.0
        return sum(w * ew for w, ew in zip(weights, exp_weights, strict=False)) / total
    raise ValueError(f"Unknown aggregation mode: {aggregation}")
