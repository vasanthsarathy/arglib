"""Core data structures."""

from .evidence import EvidenceItem
from .graph import ArgumentGraph
from .relations import Relation
from .spans import TextSpan
from .units import ArgumentUnit

__all__ = [
    "ArgumentGraph",
    "ArgumentUnit",
    "EvidenceItem",
    "Relation",
    "TextSpan",
]
