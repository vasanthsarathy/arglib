"""AI-assisted mining tools."""

from .evaluation import (
    EdgeValidationResult,
    EvidenceEvaluation,
    HeuristicEvaluator,
    score_evidence,
    validate_edges,
)
from .miner import ArgumentMiner, LongDocumentMiner, SimpleArgumentMiner
from .mining import (
    FixedWindowSplitter,
    GraphReconciler,
    MergePolicy,
    MergeResult,
    ParagraphSplitter,
    Segment,
    SimpleGraphReconciler,
    Splitter,
    token_jaccard_similarity,
)

__all__ = [
    "ArgumentMiner",
    "EdgeValidationResult",
    "EvidenceEvaluation",
    "HeuristicEvaluator",
    "FixedWindowSplitter",
    "GraphReconciler",
    "LongDocumentMiner",
    "MergePolicy",
    "MergeResult",
    "ParagraphSplitter",
    "Segment",
    "SimpleArgumentMiner",
    "SimpleGraphReconciler",
    "Splitter",
    "token_jaccard_similarity",
    "score_evidence",
    "validate_edges",
]
