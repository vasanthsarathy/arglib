"""Argument mining interfaces and baseline implementations."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from arglib.ai.llm import LLMHook
from arglib.ai.mining import (
    GraphReconciler,
    MergeResult,
    ParagraphSplitter,
    Segment,
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


class Segmenter(Protocol):
    def segment(self, text: str) -> list[str]:
        """Split raw text into chunks for mining."""


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
class HookedArgumentMiner:
    """LLM-backed miner hook with a fallback parser."""

    hook: LLMHook
    fallback: ArgumentMiner = field(default_factory=SimpleArgumentMiner)

    def parse(
        self,
        text: str,
        *,
        doc_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ArgumentGraph:
        response = self.hook.run(input=text, context=metadata)
        if response.strip():
            return _parse_json_graph(response, doc_id=doc_id) or self.fallback.parse(
                text, doc_id=doc_id, metadata=metadata
            )
        return self.fallback.parse(text, doc_id=doc_id, metadata=metadata)


@dataclass
class LongDocumentMiner:
    """Split -> mine -> reconcile pipeline for long documents."""

    miner: ArgumentMiner = field(default_factory=SimpleArgumentMiner)
    splitter: Splitter = field(default_factory=ParagraphSplitter)
    reconciler: GraphReconciler = field(default_factory=SimpleGraphReconciler)
    splitter_hook: LLMHook | None = None

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
        segments = self._split(text, metadata=metadata)
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

    def _split(
        self, text: str, *, metadata: dict[str, Any] | None = None
    ) -> list[Segment]:
        if self.splitter_hook is None:
            return self.splitter.split(text)
        response = self.splitter_hook.run(input=text, context=metadata)
        segments = _parse_segments(response)
        if segments:
            return segments
        return self.splitter.split(text)


def _offset_spans(graph: ArgumentGraph, offset: int) -> None:
    for unit in graph.units.values():
        for span in unit.spans:
            span.start += offset
            span.end += offset


def _parse_json_graph(text: str, *, doc_id: str | None = None) -> ArgumentGraph | None:
    import json

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None

    if not isinstance(payload, dict):
        return None

    graph = ArgumentGraph.from_dict(payload)
    if doc_id:
        for unit in graph.units.values():
            for span in unit.spans:
                span.doc_id = doc_id
    return graph


def _parse_segments(text: str) -> list[Segment]:
    import json

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    segments: list[Segment] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        seg_text = item.get("text")
        if not isinstance(seg_text, str):
            continue
        segments.append(
            Segment(
                id=item.get("id", f"seg-{len(segments) + 1}"),
                text=seg_text,
                start=int(item.get("start", 0)),
                end=int(item.get("end", len(seg_text))),
                metadata=dict(item.get("metadata", {})),
            )
        )
    return segments


__all__ = [
    "ArgumentMiner",
    "HookedArgumentMiner",
    "LongDocumentMiner",
    "Segmenter",
    "SimpleArgumentMiner",
]
