"""AI-assisted mining tools."""

from .evaluation import (
    EdgeValidationResult,
    EvidenceEvaluation,
    HeuristicEvaluator,
    score_evidence,
    validate_edges,
)
from .llm import LLMClient, LLMHook, NoOpLLMClient, PromptTemplate
from .miner import (
    ArgumentMiner,
    HookedArgumentMiner,
    LongDocumentMiner,
    Segmenter,
    SimpleArgumentMiner,
)
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
    "HookedArgumentMiner",
    "LLMClient",
    "LLMHook",
    "FixedWindowSplitter",
    "GraphReconciler",
    "LongDocumentMiner",
    "MergePolicy",
    "MergeResult",
    "NoOpLLMClient",
    "ParagraphSplitter",
    "PromptTemplate",
    "Segment",
    "Segmenter",
    "SimpleArgumentMiner",
    "SimpleGraphReconciler",
    "Splitter",
    "token_jaccard_similarity",
    "score_evidence",
    "validate_edges",
]
