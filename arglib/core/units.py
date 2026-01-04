"""Argument unit model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .evidence import EvidenceItem
from .spans import TextSpan


@dataclass
class ArgumentUnit:
    id: str
    text: str
    type: Literal["fact", "value", "policy", "other"] = "other"
    spans: list[TextSpan] = field(default_factory=list)
    evidence: list[EvidenceItem] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    evidence_min: float | None = None
    evidence_max: float | None = None
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
            "evidence_min": self.evidence_min,
            "evidence_max": self.evidence_max,
            "score": self.score,
            "is_axiom": self.is_axiom,
            "ignore_influence": self.ignore_influence,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArgumentUnit:
        return cls(
            id=data["id"],
            text=data["text"],
            type=data.get("type", "other"),
            spans=[TextSpan.from_dict(span) for span in data.get("spans", [])],
            evidence=[
                EvidenceItem.from_dict(item) for item in data.get("evidence", [])
            ],
            evidence_ids=list(data.get("evidence_ids", [])),
            evidence_min=data.get("evidence_min"),
            evidence_max=data.get("evidence_max"),
            score=data.get("score"),
            is_axiom=bool(data.get("is_axiom", False)),
            ignore_influence=bool(data.get("ignore_influence", False)),
            metadata=data.get("metadata", {}),
        )
