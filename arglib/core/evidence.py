"""Evidence model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .spans import TextSpan

EvidenceSource = TextSpan | dict[str, Any]


@dataclass
class EvidenceItem:
    id: str
    source: EvidenceSource
    stance: Literal["supports", "attacks", "neutral"]
    strength: float | None = None
    quality: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        if isinstance(self.source, TextSpan):
            source_payload = {
                "type": "text_span",
                "value": self.source.to_dict(),
            }
        else:
            source_payload = {
                "type": "structured",
                "value": self.source,
            }

        return {
            "id": self.id,
            "source": source_payload,
            "stance": self.stance,
            "strength": self.strength,
            "quality": self.quality,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvidenceItem:
        source_data = data["source"]
        if isinstance(source_data, dict) and source_data.get("type") == "text_span":
            source = TextSpan.from_dict(source_data["value"])
        elif isinstance(source_data, dict) and source_data.get("type") == "structured":
            source = source_data.get("value", {})
        elif isinstance(source_data, dict) and "doc_id" in source_data:
            source = TextSpan.from_dict(source_data)
        else:
            source = source_data

        return cls(
            id=data["id"],
            source=source,
            stance=data["stance"],
            strength=data.get("strength"),
            quality=data.get("quality", {}),
        )
