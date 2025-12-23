"""Relation model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional


@dataclass
class Relation:
    src: str
    dst: str
    kind: Literal["support", "attack", "undercut", "rebut"]
    weight: Optional[float] = None
    rationale: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "src": self.src,
            "dst": self.dst,
            "kind": self.kind,
            "weight": self.weight,
            "rationale": self.rationale,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Relation":
        return cls(
            src=data["src"],
            dst=data["dst"],
            kind=data["kind"],
            weight=data.get("weight"),
            rationale=data.get("rationale"),
            metadata=data.get("metadata", {}),
        )