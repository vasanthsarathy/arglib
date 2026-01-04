"""Pattern detection and gate actions for warrant-gated graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from arglib.algorithms import build_edges, find_cycles
from arglib.core import ArgumentGraph, Relation
from arglib.data import get_patterns_bank_path


@dataclass(frozen=True)
class PatternDefinition:
    pattern_id: str
    name: str
    category: str
    kind: str
    description: str
    action: str


@dataclass(frozen=True)
class PatternMatch:
    pattern_id: str
    name: str
    category: str
    kind: str
    description: str
    action: str
    nodes: list[str]
    edges: list[str]
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "name": self.name,
            "category": self.category,
            "kind": self.kind,
            "description": self.description,
            "action": self.action,
            "nodes": list(self.nodes),
            "edges": list(self.edges),
            "message": self.message,
        }


_DEFAULT_PATTERNS: dict[str, PatternDefinition] = {
    "circular_reasoning": PatternDefinition(
        pattern_id="circular_reasoning",
        name="Circular Reasoning",
        category="Structural",
        kind="fallacious",
        description="A claim ultimately supports itself through a support cycle.",
        action="disable_edge",
    ),
    "self_attack": PatternDefinition(
        pattern_id="self_attack",
        name="Self-Attack",
        category="Structural",
        kind="fallacious",
        description="A node attacks itself, contradicting its own claim.",
        action="disable_edge",
    ),
    "unsupported_conclusion": PatternDefinition(
        pattern_id="unsupported_conclusion",
        name="Unsupported Conclusion",
        category="Structural",
        kind="fallacious",
        description="A claim has no incoming support edges.",
        action="flag_node",
    ),
    "redundancy": PatternDefinition(
        pattern_id="redundancy",
        name="Redundancy",
        category="Structural",
        kind="fallacious",
        description="Multiple equivalent supports to the same conclusion.",
        action="flag_node",
    ),
    "contradiction": PatternDefinition(
        pattern_id="contradiction",
        name="Contradiction",
        category="Structural",
        kind="fallacious",
        description="Support and attack edges target the same claim from a source.",
        action="flag_edge",
    ),
    "unstated_warrant": PatternDefinition(
        pattern_id="unstated_warrant",
        name="Unstated Warrant",
        category="Substructural",
        kind="fallacious",
        description="An edge has no explicit warrants gating its inference.",
        action="disable_edge",
    ),
}


def detect_patterns(
    graph: ArgumentGraph, *, pattern_bank: dict[str, PatternDefinition] | None = None
) -> list[PatternMatch]:
    bank = pattern_bank or load_pattern_bank()
    matches: list[PatternMatch] = []

    relations = list(graph.relations)
    support_edges = [(rel.src, rel.dst, rel.kind) for rel in relations]
    support_pairs = build_edges(support_edges, kinds=["support"])

    cycles = find_cycles(graph.units.keys(), support_pairs)
    for cycle in cycles:
        if len(cycle) < 2:
            continue
        edges = _cycle_edges(cycle)
        edge_ids = _edge_ids_for_pairs(relations, edges, kind="support")
        pattern = bank["circular_reasoning"]
        matches.append(
            PatternMatch(
                pattern_id=pattern.pattern_id,
                name=pattern.name,
                category=pattern.category,
                kind=pattern.kind,
                description=pattern.description,
                action=pattern.action,
                nodes=list(cycle),
                edges=edge_ids,
                message=f"Support cycle detected across {len(cycle)} claims.",
            )
        )

    for index, rel in enumerate(relations):
        edge_id = f"e{index}"
        if rel.kind == "attack" and rel.src == rel.dst:
            pattern = bank["self_attack"]
            matches.append(
                PatternMatch(
                    pattern_id=pattern.pattern_id,
                    name=pattern.name,
                    category=pattern.category,
                    kind=pattern.kind,
                    description=pattern.description,
                    action=pattern.action,
                    nodes=[rel.src],
                    edges=[edge_id],
                    message="Attack edge targets the same claim.",
                )
            )
        if not rel.warrant_ids:
            pattern = bank["unstated_warrant"]
            matches.append(
                PatternMatch(
                    pattern_id=pattern.pattern_id,
                    name=pattern.name,
                    category=pattern.category,
                    kind=pattern.kind,
                    description=pattern.description,
                    action=pattern.action,
                    nodes=[rel.src, rel.dst],
                    edges=[edge_id],
                    message="Edge has no explicit warrants.",
                )
            )

    incoming_support: dict[str, list[int]] = {unit_id: [] for unit_id in graph.units}
    for index, rel in enumerate(relations):
        if rel.kind == "support":
            incoming_support.setdefault(rel.dst, []).append(index)

    for claim_id, support_edge_ids in incoming_support.items():
        if not support_edge_ids:
            pattern = bank["unsupported_conclusion"]
            matches.append(
                PatternMatch(
                    pattern_id=pattern.pattern_id,
                    name=pattern.name,
                    category=pattern.category,
                    kind=pattern.kind,
                    description=pattern.description,
                    action=pattern.action,
                    nodes=[claim_id],
                    edges=[],
                    message="Claim has no incoming support edges.",
                )
            )

    _detect_redundancy(graph, relations, matches, bank["redundancy"])
    _detect_contradiction(relations, matches, bank["contradiction"])

    return matches


def apply_gate_actions(graph: ArgumentGraph, matches: list[PatternMatch]) -> None:
    for match in matches:
        if match.action not in {"disable_edge", "restrict_edge"}:
            continue
        for edge_id in match.edges:
            if not edge_id.startswith("e"):
                continue
            suffix = edge_id[1:]
            if not suffix.isdigit():
                continue
            index = int(suffix)
            if index < 0 or index >= len(graph.relations):
                continue
            relation = graph.relations[index]
            if match.action == "disable_edge":
                relation.metadata["gate_disabled"] = True
                relation.metadata["gate_action"] = match.pattern_id
            elif match.action == "restrict_edge":
                relation.gate_mode = "AND"
                relation.metadata["gate_restricted"] = True
                relation.metadata["gate_action"] = match.pattern_id


def _cycle_edges(cycle: list[str]) -> list[tuple[str, str]]:
    return list(zip(cycle, cycle[1:] + [cycle[0]], strict=True))


def _edge_ids_for_pairs(
    relations: list[Relation],
    pairs: list[tuple[str, str]],
    *,
    kind: str,
) -> list[str]:
    edge_ids: list[str] = []
    for index, rel in enumerate(relations):
        if rel.kind != kind:
            continue
        if (rel.src, rel.dst) in pairs:
            edge_ids.append(f"e{index}")
    return edge_ids


def _detect_redundancy(
    graph: ArgumentGraph,
    relations: list[Relation],
    matches: list[PatternMatch],
    pattern: PatternDefinition,
) -> None:
    by_target: dict[str, dict[str, list[str]]] = {}
    for rel in relations:
        if rel.kind != "support":
            continue
        source = graph.units.get(rel.src)
        if not source:
            continue
        key = _normalize_text(source.text)
        by_target.setdefault(rel.dst, {}).setdefault(key, []).append(rel.src)

    for target, groups in by_target.items():
        for _key, sources in groups.items():
            if len(sources) < 2:
                continue
            matches.append(
                PatternMatch(
                    pattern_id=pattern.pattern_id,
                    name=pattern.name,
                    category=pattern.category,
                    kind=pattern.kind,
                    description=pattern.description,
                    action=pattern.action,
                    nodes=sorted(set(sources + [target])),
                    edges=[],
                    message=f"Redundant supports to {target}: {', '.join(sources)}.",
                )
            )


def _detect_contradiction(
    relations: list[Relation],
    matches: list[PatternMatch],
    pattern: PatternDefinition,
) -> None:
    pair_map: dict[tuple[str, str], dict[str, list[int]]] = {}
    for index, rel in enumerate(relations):
        pair_map.setdefault((rel.src, rel.dst), {}).setdefault(rel.kind, []).append(
            index
        )

    for (src, dst), kinds in pair_map.items():
        if "support" in kinds and "attack" in kinds:
            edges = [f"e{idx}" for idx in kinds["support"] + kinds["attack"]]
            matches.append(
                PatternMatch(
                    pattern_id=pattern.pattern_id,
                    name=pattern.name,
                    category=pattern.category,
                    kind=pattern.kind,
                    description=pattern.description,
                    action=pattern.action,
                    nodes=[src, dst],
                    edges=edges,
                    message="Same source both supports and attacks the target.",
                )
            )


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def load_pattern_bank() -> dict[str, PatternDefinition]:
    path = get_patterns_bank_path()
    if not path:
        return dict(_DEFAULT_PATTERNS)
    try:
        import yaml  # type: ignore
    except Exception:
        return dict(_DEFAULT_PATTERNS)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return dict(_DEFAULT_PATTERNS)
    entries = data.get("patterns", []) if isinstance(data, dict) else []
    bank: dict[str, PatternDefinition] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        pattern_id = entry.get("id")
        if not pattern_id:
            continue
        bank[pattern_id] = PatternDefinition(
            pattern_id=pattern_id,
            name=str(entry.get("name") or pattern_id),
            category=str(entry.get("category") or "Uncategorized"),
            kind=str(entry.get("kind") or "fallacious"),
            description=str(entry.get("description") or ""),
            action=str(entry.get("action") or "flag_edge"),
        )
    for key, definition in _DEFAULT_PATTERNS.items():
        bank.setdefault(key, definition)
    return bank
