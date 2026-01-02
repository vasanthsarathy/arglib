"""Core data structures."""

from .bundles import ArgumentBundle, ArgumentBundleGraph
from .evidence import EvidenceCard, EvidenceItem, SupportingDocument
from .graph import ArgumentGraph
from .relations import Relation
from .spans import TextSpan
from .units import ArgumentUnit
from .warrants import Warrant, WarrantAttack

__all__ = [
    "ArgumentGraph",
    "ArgumentBundle",
    "ArgumentBundleGraph",
    "ArgumentUnit",
    "EvidenceCard",
    "EvidenceItem",
    "Relation",
    "SupportingDocument",
    "TextSpan",
    "Warrant",
    "WarrantAttack",
]
