"""Argument mining interfaces and baseline implementations."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from arglib.ai.mining import (
    GraphReconciler,
    MergeResult,
    ParagraphSplitter,
    SimpleGraphReconciler,
    Splitter,
)
from arglib.core import ArgumentGraph, TextSpan


class ArgumentMiner(Protocol):
    def parse(
        self,
        text: str,
        *,
        doc_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ArgumentGraph:
        """Parse text into an argument graph."""


@dataclass
class SimpleArgumentMiner:
    """Baseline miner that converts sentences into isolated claims."""

    sentence_pattern: re.Pattern[str] = re.compile(r"[^.!?]+[.!?]?")

    def parse(
        self,
        text: str,
        *,
        doc_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ArgumentGraph:
        graph = ArgumentGraph.new()
        if metadata:
            graph.metadata.update(metadata)
        source_id = doc_id or "document"

        for match in self.sentence_pattern.finditer(text):
            sentence = match.group(0).strip()
            if not sentence:
                continue
            start = match.start()
            end = match.end()
            span = TextSpan(
                doc_id=source_id,
                start=start,
                end=end,
                text=sentence,
                modality="text",
            )
            graph.add_claim(sentence, spans=[span])
        return graph


@dataclass
class LongDocumentMiner:
    """Split -> mine -> reconcile pipeline for long documents."""

    miner: ArgumentMiner = field(default_factory=SimpleArgumentMiner)
    splitter: Splitter = field(default_factory=ParagraphSplitter)
    reconciler: GraphReconciler = field(default_factory=SimpleGraphReconciler)

    def parse(
        self,
        text: str,
        *,
        doc_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ArgumentGraph:
        result = self.parse_with_segments(text, doc_id=doc_id, metadata=metadata)
        return result.graph

    def parse_with_segments(
        self,
        text: str,
        *,
        doc_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MergeResult:
        segments = self.splitter.split(text)
        graphs: list[ArgumentGraph] = []
        for segment in segments:
            graph = self.miner.parse(
                segment.text,
                doc_id=doc_id,
                metadata={"segment_id": segment.id, **(metadata or {})},
            )
            if segment.start:
                _offset_spans(graph, segment.start)
            graphs.append(graph)
        return self.reconciler.reconcile(segments, graphs)


def _offset_spans(graph: ArgumentGraph, offset: int) -> None:
    for unit in graph.units.values():
        for span in unit.spans:
            span.start += offset
            span.end += offset


__all__ = [
    "ArgumentMiner",
    "LongDocumentMiner",
    "SimpleArgumentMiner",
]
