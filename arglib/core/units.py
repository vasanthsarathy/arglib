"""Argument unit model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal

from .evidence import EvidenceItem
from .spans import TextSpan


@dataclass
class ArgumentUnit:
    id: str
    text: str
    type: Literal["fact", "value", "policy", "other"] = "other"
    spans: List[TextSpan] = field(default_factory=list)
    evidence: List[EvidenceItem] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "type": self.type,
            "spans": [span.to_dict() for span in self.spans],
            "evidence": [item.to_dict() for item in self.evidence],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArgumentUnit":
        return cls(
            id=data["id"],
            text=data["text"],
            type=data.get("type", "other"),
            spans=[TextSpan.from_dict(span) for span in data.get("spans", [])],
            evidence=[EvidenceItem.from_dict(item) for item in data.get("evidence", [])],
            metadata=data.get("metadata", {}),
        )