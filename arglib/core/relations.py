"""Relation model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class Relation:
    src: str
    dst: str
    kind: Literal["support", "attack", "undercut", "rebut"]
    warrant_ids: list[str] = field(default_factory=list)
    gate_mode: Literal["AND", "OR"] = "OR"
    weight: float | None = None
    rationale: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "src": self.src,
            "dst": self.dst,
            "kind": self.kind,
            "warrant_ids": list(self.warrant_ids),
            "gate_mode": self.gate_mode,
            "weight": self.weight,
            "rationale": self.rationale,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Relation:
        return cls(
            src=data["src"],
            dst=data["dst"],
            kind=data["kind"],
            warrant_ids=list(data.get("warrant_ids", [])),
            gate_mode=data.get("gate_mode", "OR"),
            weight=data.get("weight"),
            rationale=data.get("rationale"),
            metadata=data.get("metadata", {}),
        )
