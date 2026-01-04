"""Warrant model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .evidence import EvidenceItem
from .spans import TextSpan


@dataclass
class Warrant:
    id: str
    text: str
    type: Literal["fact", "value", "policy", "other"] = "other"
    spans: list[TextSpan] = field(default_factory=list)
    evidence: list[EvidenceItem] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    score: float | None = None
    is_axiom: bool = False
    ignore_influence: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "type": self.type,
            "spans": [span.to_dict() for span in self.spans],
            "evidence": [item.to_dict() for item in self.evidence],
            "evidence_ids": list(self.evidence_ids),
            "score": self.score,
            "is_axiom": self.is_axiom,
            "ignore_influence": self.ignore_influence,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Warrant:
        return cls(
            id=data["id"],
            text=data["text"],
            type=data.get("type", "other"),
            spans=[TextSpan.from_dict(span) for span in data.get("spans", [])],
            evidence=[
                EvidenceItem.from_dict(item) for item in data.get("evidence", [])
            ],
            evidence_ids=list(data.get("evidence_ids", [])),
            score=data.get("score"),
            is_axiom=bool(data.get("is_axiom", False)),
            ignore_influence=bool(data.get("ignore_influence", False)),
            metadata=data.get("metadata", {}),
        )


@dataclass
class WarrantAttack:
    src: str
    warrant_id: str
    kind: Literal["attack"] = "attack"
    rationale: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "src": self.src,
            "warrant_id": self.warrant_id,
            "kind": self.kind,
            "rationale": self.rationale,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WarrantAttack:
        return cls(
            src=data["src"],
            warrant_id=data["warrant_id"],
            kind=data.get("kind", "attack"),
            rationale=data.get("rationale"),
            metadata=data.get("metadata", {}),
        )


__all__ = ["Warrant", "WarrantAttack"]
