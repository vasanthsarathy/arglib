"""Text span model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class TextSpan:
    doc_id: str
    start: int
    end: int
    text: str
    page: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    modality: Literal["text", "pdf", "image", "video", "audio"] = "text"

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "page": self.page,
            "bbox": self.bbox,
            "modality": self.modality,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TextSpan:
        return cls(
            doc_id=data["doc_id"],
            start=data["start"],
            end=data["end"],
            text=data["text"],
            page=data.get("page"),
            bbox=tuple(data["bbox"]) if data.get("bbox") is not None else None,
            modality=data.get("modality", "text"),
        )
