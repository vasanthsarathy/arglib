"""Argument bundle abstractions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .relations import Relation


@dataclass
class ArgumentBundle:
    id: str
    units: list[str]
    relations: list[Relation] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "units": list(self.units),
            "relations": [relation.to_dict() for relation in self.relations],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArgumentBundle:
        return cls(
            id=data["id"],
            units=list(data.get("units", [])),
            relations=[Relation.from_dict(item) for item in data.get("relations", [])],
            metadata=data.get("metadata", {}),
        )


@dataclass
class ArgumentBundleGraph:
    bundles: dict[str, ArgumentBundle] = field(default_factory=dict)
    relations: list[Relation] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bundles": {
                bundle_id: bundle.to_dict()
                for bundle_id, bundle in self.bundles.items()
            },
            "relations": [relation.to_dict() for relation in self.relations],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArgumentBundleGraph:
        bundles = {
            bundle_id: ArgumentBundle.from_dict(bundle)
            for bundle_id, bundle in data.get("bundles", {}).items()
        }
        relations = [Relation.from_dict(item) for item in data.get("relations", [])]
        return cls(
            bundles=bundles,
            relations=relations,
            metadata=data.get("metadata", {}),
        )
