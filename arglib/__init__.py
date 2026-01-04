"""ArgLib package."""

from arglib.core import (
    ArgumentGraph,
    ArgumentUnit,
    EvidenceCard,
    EvidenceItem,
    Relation,
    SupportingDocument,
    TextSpan,
)

__all__ = [
    "ArgumentGraph",
    "ArgumentUnit",
    "EvidenceCard",
    "EvidenceItem",
    "Relation",
    "SupportingDocument",
    "TextSpan",
    "__version__",
]

__version__ = "0.1.10"
