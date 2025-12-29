"""Long-document mining interfaces and basic implementations."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from arglib.core import ArgumentGraph, ArgumentUnit, Relation

RelationKind = Literal["support", "attack", "undercut", "rebut"]
RelationKey = tuple[str, str, RelationKind]


@dataclass(frozen=True)
class Segment:
    id: str
    text: str
    start: int
    end: int
    metadata: dict[str, Any] = field(default_factory=dict)


def token_jaccard_similarity(left: str, right: str) -> float:
    left_tokens = set(re.findall(r"[a-zA-Z0-9]+", left.lower()))
    right_tokens = set(re.findall(r"[a-zA-Z0-9]+", right.lower()))
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    intersection = left_tokens & right_tokens
    union = left_tokens | right_tokens
    return len(intersection) / len(union)


class Splitter(Protocol):
    def split(self, text: str) -> list[Segment]:
        """Split a document into ordered segments with offsets."""


class ParagraphSplitter:
    """Split text by blank lines while preserving offsets."""

    def __init__(self, *, min_chars: int = 1) -> None:
        self.min_chars = min_chars

    def split(self, text: str) -> list[Segment]:
        segments: list[Segment] = []
        pattern = re.compile(r"(?:^|\n\s*\n)(?P<chunk>[^\n].*?)(?=\n\s*\n|$)", re.S)
        for match in pattern.finditer(text):
            chunk = match.group("chunk")
            if len(chunk.strip()) < self.min_chars:
                continue
            start = match.start("chunk")
            end = match.end("chunk")
            segments.append(
                Segment(
                    id=f"seg-{len(segments) + 1}",
                    text=chunk,
                    start=start,
                    end=end,
                )
            )
        if not segments and text.strip():
            segments.append(Segment(id="seg-1", text=text, start=0, end=len(text)))
        return segments


class FixedWindowSplitter:
    """Split text into fixed-length windows with optional overlap."""

    def __init__(self, *, window_size: int = 1000, overlap: int = 0) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if overlap < 0 or overlap >= window_size:
            raise ValueError("overlap must be in [0, window_size)")
        self.window_size = window_size
        self.overlap = overlap

    def split(self, text: str) -> list[Segment]:
        segments: list[Segment] = []
        step = self.window_size - self.overlap
        for start in range(0, max(len(text), 1), step):
            end = min(start + self.window_size, len(text))
            chunk = text[start:end]
            if not chunk.strip():
                continue
            segments.append(
                Segment(
                    id=f"seg-{len(segments) + 1}",
                    text=chunk,
                    start=start,
                    end=end,
                )
            )
            if end >= len(text):
                break
        return segments


@dataclass
class MergePolicy:
    unit_key: Callable[[ArgumentUnit], str] | None = None
    relation_aggregation: Literal["sum", "max", "mean"] = "sum"
    similarity_fn: Callable[[str, str], float] | None = None
    similarity_threshold: float = 0.9
    clamp_weights: bool = True
    weight_min: float = -1.0
    weight_max: float = 1.0

    def key_for_unit(self, unit: ArgumentUnit) -> str:
        if self.unit_key is not None:
            return self.unit_key(unit)
        return unit.text.strip().lower()

    def match_existing_key(self, key: str, existing: Sequence[str]) -> str | None:
        if not self.similarity_fn:
            return None
        best_key: str | None = None
        best_score = 0.0
        for candidate in existing:
            score = self.similarity_fn(key, candidate)
            if score > best_score:
                best_score = score
                best_key = candidate
        if best_key is not None and best_score >= self.similarity_threshold:
            return best_key
        return None

    def aggregate_weights(self, weights: list[float]) -> float:
        if not weights:
            return 0.0
        if self.relation_aggregation == "max":
            value = max(weights, key=abs)
        elif self.relation_aggregation == "mean":
            value = sum(weights) / len(weights)
        else:
            value = sum(weights)
        if self.clamp_weights:
            value = max(self.weight_min, min(self.weight_max, value))
        return value


@dataclass
class MergeResult:
    graph: ArgumentGraph
    unit_segments: dict[str, list[str]]
    relation_sources: dict[RelationKey, list[dict[str, Any]]]


class GraphReconciler(Protocol):
    def reconcile(
        self, segments: Sequence[Segment], graphs: Sequence[ArgumentGraph]
    ) -> MergeResult:
        """Merge per-segment graphs into a unified graph."""


class SimpleGraphReconciler:
    """Merge graphs by exact text match and aggregate edges."""

    def __init__(self, policy: MergePolicy | None = None) -> None:
        self.policy = policy or MergePolicy()

    def reconcile(
        self, segments: Sequence[Segment], graphs: Sequence[ArgumentGraph]
    ) -> MergeResult:
        if len(segments) != len(graphs):
            raise ValueError("segments and graphs must be the same length")

        merged = ArgumentGraph.new()
        merged.metadata["segments"] = [
            {"id": segment.id, "start": segment.start, "end": segment.end}
            for segment in segments
        ]

        unit_segments: dict[str, list[str]] = {}
        unit_key_to_id: dict[str, str] = {}
        relation_sources: dict[RelationKey, list[dict[str, Any]]] = {}
        relation_weights: dict[RelationKey, list[float]] = {}
        relation_map: dict[RelationKey, Relation] = {}
        conflicts: list[dict[str, Any]] = []

        for segment, graph in zip(segments, graphs, strict=True):
            for doc_id, doc in graph.supporting_documents.items():
                if doc_id in merged.supporting_documents:
                    if merged.supporting_documents[doc_id].to_dict() != doc.to_dict():
                        conflicts.append({"supporting_document_id": doc_id})
                    continue
                merged.supporting_documents[doc_id] = doc

            for card_id, card in graph.evidence_cards.items():
                if card_id in merged.evidence_cards:
                    if merged.evidence_cards[card_id].to_dict() != card.to_dict():
                        conflicts.append({"evidence_card_id": card_id})
                    continue
                merged.evidence_cards[card_id] = card

            unit_map: dict[str, str] = {}
            for unit_id, unit in graph.units.items():
                key = self.policy.key_for_unit(unit)
                if key not in unit_key_to_id:
                    match = self.policy.match_existing_key(
                        key, list(unit_key_to_id.keys())
                    )
                else:
                    match = None

                if match is not None:
                    new_id = unit_key_to_id[match]
                    existing = merged.units[new_id]
                    existing.spans.extend(unit.spans)
                    existing.evidence.extend(unit.evidence)
                    for evidence_id in unit.evidence_ids:
                        if evidence_id not in existing.evidence_ids:
                            existing.evidence_ids.append(evidence_id)
                    existing.metadata.setdefault("source_unit_ids", []).append(unit_id)
                    similarity_fn = self.policy.similarity_fn
                    if similarity_fn is not None:
                        existing.metadata.setdefault("merge_matches", []).append(
                            {"key": key, "score": similarity_fn(key, match)}
                        )
                    if segment.id not in unit_segments[new_id]:
                        unit_segments[new_id].append(segment.id)
                    if unit.evidence_min is not None:
                        existing.evidence_min = (
                            unit.evidence_min
                            if existing.evidence_min is None
                            else min(existing.evidence_min, unit.evidence_min)
                        )
                    if unit.evidence_max is not None:
                        existing.evidence_max = (
                            unit.evidence_max
                            if existing.evidence_max is None
                            else max(existing.evidence_max, unit.evidence_max)
                        )
                    unit_map[unit_id] = new_id
                    continue

                if key not in unit_key_to_id:
                    new_id = merged.add_claim(
                        unit.text,
                        type=unit.type,
                        spans=list(unit.spans),
                        evidence=list(unit.evidence),
                        metadata=dict(unit.metadata),
                        evidence_ids=list(unit.evidence_ids),
                        evidence_min=unit.evidence_min,
                        evidence_max=unit.evidence_max,
                    )
                    merged.units[new_id].metadata.setdefault(
                        "source_unit_ids", []
                    ).append(unit_id)
                    unit_key_to_id[key] = new_id
                    unit_segments[new_id] = [segment.id]
                else:
                    new_id = unit_key_to_id[key]
                    existing = merged.units[new_id]
                    existing.spans.extend(unit.spans)
                    existing.evidence.extend(unit.evidence)
                    for evidence_id in unit.evidence_ids:
                        if evidence_id not in existing.evidence_ids:
                            existing.evidence_ids.append(evidence_id)
                    existing.metadata.setdefault("source_unit_ids", []).append(unit_id)
                    if segment.id not in unit_segments[new_id]:
                        unit_segments[new_id].append(segment.id)
                    if unit.evidence_min is not None:
                        existing.evidence_min = (
                            unit.evidence_min
                            if existing.evidence_min is None
                            else min(existing.evidence_min, unit.evidence_min)
                        )
                    if unit.evidence_max is not None:
                        existing.evidence_max = (
                            unit.evidence_max
                            if existing.evidence_max is None
                            else max(existing.evidence_max, unit.evidence_max)
                        )
                unit_map[unit_id] = new_id

            for relation in graph.relations:
                src = unit_map.get(relation.src)
                dst = unit_map.get(relation.dst)
                if src is None or dst is None:
                    continue
                relation_key: RelationKey = (src, dst, relation.kind)
                weight = 1.0 if relation.weight is None else relation.weight
                relation_weights.setdefault(relation_key, []).append(weight)
                relation_sources.setdefault(relation_key, []).append(
                    {"segment_id": segment.id, "relation": relation.to_dict()}
                )
                if relation_key not in relation_map:
                    merged_relation = Relation(
                        src=src,
                        dst=dst,
                        kind=relation.kind,
                        weight=weight,
                        rationale=relation.rationale,
                        metadata=dict(relation.metadata),
                    )
                    merged_relation.metadata.setdefault("sources", []).append(
                        {"segment_id": segment.id}
                    )
                    merged.relations.append(merged_relation)
                    relation_map[relation_key] = merged_relation
                else:
                    relation_map[relation_key].metadata.setdefault(
                        "sources", []
                    ).append({"segment_id": segment.id})

        for relation_key, relation in relation_map.items():
            relation.weight = self.policy.aggregate_weights(
                relation_weights.get(relation_key, [])
            )

        if conflicts:
            merged.metadata["merge_conflicts"] = conflicts

        return MergeResult(
            graph=merged,
            unit_segments=unit_segments,
            relation_sources=relation_sources,
        )
