"""Argument mining interfaces and baseline implementations."""

from __future__ import annotations

import asyncio
import inspect
import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

from arglib.ai.llm import AsyncLLMHook, LLMHook
from arglib.ai.mining import (
    GraphReconciler,
    MergeResult,
    ParagraphSplitter,
    Segment,
    SimpleGraphReconciler,
    Splitter,
)
from arglib.core import ArgumentGraph, TextSpan
from arglib.io import validate_graph_dict


class ArgumentMiner(Protocol):
    def parse(
        self,
        text: str,
        *,
        doc_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ArgumentGraph:
        """Parse text into an argument graph."""


class AsyncArgumentMiner(Protocol):
    async def parse(
        self,
        text: str,
        *,
        doc_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ArgumentGraph:
        """Parse text into an argument graph asynchronously."""


@dataclass
class AsyncArgumentMinerAdapter:
    miner: ArgumentMiner

    async def parse(
        self,
        text: str,
        *,
        doc_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ArgumentGraph:
        return await asyncio.to_thread(
            self.miner.parse, text, doc_id=doc_id, metadata=metadata
        )


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
        graph, errors = _parse_json_graph(response, doc_id=doc_id)
        if graph is not None and not errors:
            return graph
        fallback = self.fallback.parse(text, doc_id=doc_id, metadata=metadata)
        if errors:
            fallback.metadata.setdefault("llm_parse_errors", errors)
            fallback.metadata.setdefault("llm_raw_response", response[:500])
        return fallback


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


@dataclass
class AsyncLongDocumentMiner:
    miner: AsyncArgumentMiner | ArgumentMiner
    splitter: Splitter = field(default_factory=ParagraphSplitter)
    reconciler: GraphReconciler = field(default_factory=SimpleGraphReconciler)
    splitter_hook: AsyncLLMHook | None = None

    async def parse(
        self,
        text: str,
        *,
        doc_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ArgumentGraph:
        result = await self.parse_with_segments(text, doc_id=doc_id, metadata=metadata)
        return result.graph

    async def parse_with_segments(
        self,
        text: str,
        *,
        doc_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MergeResult:
        segments = await self._split(text, metadata=metadata)
        tasks = [
            self._parse_segment(
                segment,
                doc_id=doc_id,
                metadata=metadata,
            )
            for segment in segments
        ]
        graphs = list(await asyncio.gather(*tasks))
        for segment, graph in zip(segments, graphs, strict=True):
            if segment.start:
                _offset_spans(graph, segment.start)
        return self.reconciler.reconcile(segments, graphs)

    async def _split(
        self, text: str, *, metadata: dict[str, Any] | None = None
    ) -> list[Segment]:
        if self.splitter_hook is None:
            return self.splitter.split(text)
        response = await self.splitter_hook.run(input=text, context=metadata)
        segments = _parse_segments(response)
        if segments:
            return segments
        return self.splitter.split(text)

    async def _parse_segment(
        self,
        segment: Segment,
        *,
        doc_id: str | None,
        metadata: dict[str, Any] | None,
    ) -> ArgumentGraph:
        parse = self.miner.parse
        meta = {"segment_id": segment.id, **(metadata or {})}
        if inspect.iscoroutinefunction(parse):
            async_parse = cast(Any, parse)
            return await async_parse(segment.text, doc_id=doc_id, metadata=meta)
        sync_parse = cast(Any, parse)
        return await asyncio.to_thread(
            sync_parse, segment.text, doc_id=doc_id, metadata=meta
        )


def _offset_spans(graph: ArgumentGraph, offset: int) -> None:
    for unit in graph.units.values():
        for span in unit.spans:
            span.start += offset
            span.end += offset


def _parse_json_graph(
    text: str, *, doc_id: str | None = None
) -> tuple[ArgumentGraph | None, list[str]]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, [f"invalid json: {exc}"]

    if not isinstance(payload, dict):
        return None, ["payload must be an object"]

    errors = validate_graph_dict(payload)
    if errors:
        return None, errors

    graph = ArgumentGraph.from_dict(payload)
    if doc_id:
        for unit in graph.units.values():
            for span in unit.spans:
                span.doc_id = doc_id
    return graph, []


def _parse_segments(text: str) -> list[Segment]:
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
    "AsyncArgumentMiner",
    "AsyncArgumentMinerAdapter",
    "AsyncLongDocumentMiner",
    "HookedArgumentMiner",
    "LongDocumentMiner",
    "Segmenter",
    "SimpleArgumentMiner",
]
