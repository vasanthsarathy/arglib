"""Evidence model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .spans import TextSpan

EvidenceSource = TextSpan | dict[str, Any]


@dataclass
class SupportingDocument:
    id: str
    name: str
    type: str
    url: str
    trust: float | None = None
    size: float | None = None
    upload_date: str | None = None
    uploader: str | None = None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "url": self.url,
            "trust": self.trust,
            "size": self.size,
            "upload_date": self.upload_date,
            "uploader": self.uploader,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SupportingDocument:
        return cls(
            id=data["id"],
            name=data["name"],
            type=data["type"],
            url=data["url"],
            trust=data.get("trust"),
            size=data.get("size"),
            upload_date=data.get("upload_date"),
            uploader=data.get("uploader"),
            metadata=data.get("metadata"),
        )


@dataclass
class EvidenceCard:
    id: str
    title: str
    supporting_doc_id: str
    excerpt: str
    confidence: float
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "supporting_doc_id": self.supporting_doc_id,
            "excerpt": self.excerpt,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvidenceCard:
        return cls(
            id=data["id"],
            title=data["title"],
            supporting_doc_id=data["supporting_doc_id"],
            excerpt=data["excerpt"],
            confidence=data["confidence"],
            metadata=data.get("metadata"),
        )


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
