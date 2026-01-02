"""Argument mining interfaces and baseline implementations."""

from __future__ import annotations

import asyncio
import inspect
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, cast

from arglib.ai.llm import AsyncLLMHook, LLMHook, PromptTemplate
from arglib.ai.mining import (
    GraphReconciler,
    MergeResult,
    ParagraphSplitter,
    Segment,
    SimpleGraphReconciler,
    Splitter,
    token_jaccard_similarity,
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


ARGUMENT_MINING_TEMPLATE = PromptTemplate(
    system=(
        "You are an expert argument mining assistant. Extract claims and "
        "relationships from text. Return JSON only, matching the schema exactly."
    ),
    user=(
        "Text:\n{input}\n\n"
        "Return a JSON object with this schema:\n"
        "{{\n"
        '  "units": {{\n'
        '    "c1": {{"id": "c1", "text": "...", "type": "fact", '
        '"spans": [], "evidence": [], "evidence_ids": [], "metadata": {{}}}},\n'
        '    "c2": {{"id": "c2", "text": "...", "type": "value", '
        '"spans": [], "evidence": [], "evidence_ids": [], "metadata": {{}}}}\n'
        "  }},\n"
        '  "relations": [\n'
        '    {{"src": "c1", "dst": "c2", "kind": "support", "weight": 0.6}}\n'
        "  ],\n"
        '  "metadata": {{"source": "llm"}}\n'
        "}}\n\n"
        "Rules:\n"
        "- Use relation kinds: support, attack, undercut, rebut.\n"
        "- Keep ids stable (c1, c2, ...). All relation src/dst must exist.\n"
        "- Include only claims stated or strongly implied by the text.\n"
        "- Return JSON only. No markdown, no commentary.\n"
    ),
)


CLAIM_EXTRACTION_TEMPLATE = PromptTemplate(
    system=(
        "You are an expert argument mining assistant. Extract atomic claims "
        "from text. Each claim should state a single point. Return JSON only."
    ),
    user=(
        "Text:\n{input}\n\n"
        "Return JSON with this schema:\n"
        "{{\n"
        '  "claims": [\n'
        '    {{"text": "...", "type": "fact"}},\n'
        '    {{"text": "...", "type": "value"}}\n'
        "  ]\n"
        "}}\n\n"
        "Rules:\n"
        "- Split compound statements into multiple atomic claims.\n"
        "- Types: fact, value, policy, other.\n"
        "- Only include claims stated or strongly implied.\n"
        "- Return JSON only. No markdown or commentary.\n"
    ),
)


RELATION_INFERENCE_TEMPLATE = PromptTemplate(
    system=(
        "You are an expert argument mining assistant. Infer the intended "
        "support or attack relations between claims. Return JSON only."
    ),
    user=(
        "Claims:\n{claims}\n\n"
        "Return JSON with this schema:\n"
        "{{\n"
        '  "relations": [\n'
        '    {{"src": "c1", "dst": "c2", "kind": "support"}},\n'
        '    {{"src": "c3", "dst": "c2", "kind": "attack"}}\n'
        "  ]\n"
        "}}\n\n"
        "Rules:\n"
        "- Use kinds: support, attack, undercut, rebut.\n"
        "- Only include relations intended by the author.\n"
        "- Relations can span paragraphs.\n"
        "- Return JSON only. No markdown or commentary.\n"
    ),
)


def build_argument_mining_pipeline(
    client,
    *,
    relation_client=None,
    splitter: Splitter | None = None,
    similarity_threshold: float = 0.92,
) -> ArgumentMiningPipeline:
    claim_hook = LLMHook(client=client, template=CLAIM_EXTRACTION_TEMPLATE)
    relation_hook = LLMHook(
        client=relation_client or client, template=RELATION_INFERENCE_TEMPLATE
    )
    return ArgumentMiningPipeline(
        claim_hook=claim_hook,
        relation_hook=relation_hook,
        splitter=splitter or ParagraphSplitter(),
        similarity_threshold=similarity_threshold,
    )


def build_argument_mining_hook(client) -> LLMHook:
    return LLMHook(client=client, template=ARGUMENT_MINING_TEMPLATE)


def build_argument_miner(
    client, *, fallback: ArgumentMiner | None = None
) -> HookedArgumentMiner:
    return HookedArgumentMiner(
        hook=build_argument_mining_hook(client),
        fallback=fallback or SimpleArgumentMiner(),
    )


@dataclass
class ExtractedClaim:
    id: str
    text: str
    type: str
    segment_id: str
    span_start: int | None
    span_end: int | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ArgumentMiningPipeline:
    claim_hook: LLMHook
    relation_hook: LLMHook | None = None
    splitter: Splitter = field(default_factory=ParagraphSplitter)
    similarity_fn: Callable[[str, str], float] = token_jaccard_similarity
    similarity_threshold: float = 0.92
    allow_fallback: bool = True
    fallback: ArgumentMiner | None = None

    def parse(
        self,
        text: str,
        *,
        doc_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ArgumentGraph:
        trace: list[str] = []
        segments = self.splitter.split(text)
        trace.append(f"segments: {len(segments)}")

        claims: list[ExtractedClaim] = []
        for segment in segments:
            segment_claims = self._extract_claims(segment, doc_id, trace)
            claims.extend(segment_claims)

        if not claims and self.allow_fallback:
            trace.append("no claims from llm; fallback miner")
            return self._fallback_graph(text, doc_id, metadata, trace)

        merged_claims, merge_meta = self._merge_claims(claims)
        trace.append(f"claims: {len(merged_claims)} (merged {merge_meta['merged']})")

        relations = self._infer_relations(merged_claims, trace)
        trace.append(f"relations: {len(relations)}")

        graph = ArgumentGraph.new()
        if metadata:
            graph.metadata.update(metadata)
        graph.metadata.setdefault("mining", {})
        graph.metadata["mining"].update(
            {
                "trace": trace,
                "segments": [segment.id for segment in segments],
                "merge_summary": merge_meta,
            }
        )
        for claim in merged_claims:
            spans = []
            if claim.span_start is not None and claim.span_end is not None:
                spans.append(
                    TextSpan(
                        doc_id=doc_id or "document",
                        start=claim.span_start,
                        end=claim.span_end,
                        text=claim.text,
                        modality="text",
                    )
                )
            claim_type = (
                claim.type if claim.type in {"fact", "value", "policy"} else "other"
            )
            claim_type_literal = cast(
                Literal["fact", "value", "policy", "other"], claim_type
            )
            graph.add_claim(
                claim.text,
                claim_id=claim.id,
                type=claim_type_literal,
                spans=spans,
                metadata=dict(claim.metadata),
            )
        for relation in relations:
            graph.add_relation(
                relation["src"],
                relation["dst"],
                kind=relation["kind"],
                weight=relation.get("weight"),
                rationale=relation.get("rationale"),
                metadata=dict(relation.get("metadata", {})),
            )
        return graph

    def _extract_claims(
        self,
        segment: Segment,
        doc_id: str | None,
        trace: list[str],
    ) -> list[ExtractedClaim]:
        response = self.claim_hook.run(input=segment.text, context=None)
        parsed = _parse_json_payload(response)
        claims_payload = parsed.get("claims") if isinstance(parsed, dict) else None
        if not isinstance(claims_payload, list):
            trace.append(f"{segment.id}: claim parse failed")
            return []
        results: list[ExtractedClaim] = []
        for item in claims_payload:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            claim_type = str(item.get("type", "other")).lower()
            claim_id = f"c{len(results) + 1}"
            span_start, span_end = _locate_span(segment, text)
            results.append(
                ExtractedClaim(
                    id=claim_id,
                    text=text,
                    type=claim_type,
                    segment_id=segment.id,
                    span_start=span_start,
                    span_end=span_end,
                    metadata={"segment_id": segment.id},
                )
            )
        trace.append(f"{segment.id}: claims {len(results)}")
        return results

    def _merge_claims(
        self, claims: list[ExtractedClaim]
    ) -> tuple[list[ExtractedClaim], dict[str, Any]]:
        merged: list[ExtractedClaim] = []
        mapping: dict[str, str] = {}
        merged_count = 0
        for claim in claims:
            key = claim.text.strip().lower()
            match = None
            for existing in merged:
                score = self.similarity_fn(key, existing.text.strip().lower())
                if score >= self.similarity_threshold:
                    match = existing
                    break
            if match:
                mapping[claim.id] = match.id
                merged_count += 1
                match.metadata.setdefault("merge_sources", []).append(
                    {"segment_id": claim.segment_id, "text": claim.text}
                )
                continue
            new_id = f"c{len(merged) + 1}"
            mapping[claim.id] = new_id
            merged.append(
                ExtractedClaim(
                    id=new_id,
                    text=claim.text,
                    type=claim.type,
                    segment_id=claim.segment_id,
                    span_start=claim.span_start,
                    span_end=claim.span_end,
                    metadata=dict(claim.metadata),
                )
            )
        return merged, {"merged": merged_count, "mapping": mapping}

    def _infer_relations(
        self, claims: list[ExtractedClaim], trace: list[str]
    ) -> list[dict[str, Any]]:
        if not claims:
            return []
        hook = self.relation_hook or self.claim_hook
        claim_lines = "\n".join(
            f"- {claim.id} ({claim.type}) [{claim.segment_id}]: {claim.text}"
            for claim in claims
        )
        response = hook.run(input="relations", context={"claims": claim_lines})
        parsed = _parse_json_payload(response)
        relations_payload = (
            parsed.get("relations") if isinstance(parsed, dict) else None
        )
        if not isinstance(relations_payload, list):
            trace.append("relations parse failed")
            return []
        relations: list[dict[str, Any]] = []
        valid_ids = {claim.id for claim in claims}
        for item in relations_payload:
            if not isinstance(item, dict):
                continue
            src = str(item.get("src", "")).strip()
            dst = str(item.get("dst", "")).strip()
            kind = str(item.get("kind", "")).strip().lower()
            if not src or not dst or src == dst:
                continue
            if src not in valid_ids or dst not in valid_ids:
                continue
            if kind not in {"support", "attack", "undercut", "rebut"}:
                continue
            relations.append(
                {
                    "src": src,
                    "dst": dst,
                    "kind": kind,
                    "rationale": item.get("rationale"),
                    "metadata": {},
                }
            )
        return relations

    def _fallback_graph(
        self,
        text: str,
        doc_id: str | None,
        metadata: dict[str, Any] | None,
        trace: list[str],
    ) -> ArgumentGraph:
        fallback = self.fallback or SimpleArgumentMiner()
        graph = fallback.parse(text, doc_id=doc_id, metadata=metadata)
        graph.metadata.setdefault("mining", {})
        graph.metadata["mining"]["trace"] = trace
        graph.metadata["mining"]["llm_fallback"] = True
        return graph


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
    raw = text.strip()
    payload = _parse_json_payload(raw)
    if payload is None:
        return None, ["invalid json: unable to parse payload"]

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


def _parse_json_payload(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    if "```" in text:
        cleaned = text.replace("```json", "```").replace("```JSON", "```")
        parts = cleaned.split("```")
        for part in parts:
            snippet = part.strip()
            if not snippet:
                continue
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                continue
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and start < end:
        snippet = text[start : end + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            return {}
    return {}


def _locate_span(segment: Segment, claim_text: str) -> tuple[int | None, int | None]:
    lowered = segment.text.lower()
    claim_lower = claim_text.lower()
    idx = lowered.find(claim_lower)
    if idx == -1:
        return None, None
    start = segment.start + idx
    end = start + len(claim_text)
    return start, end


__all__ = [
    "ArgumentMiner",
    "ArgumentMiningPipeline",
    "AsyncArgumentMiner",
    "AsyncArgumentMinerAdapter",
    "AsyncLongDocumentMiner",
    "CLAIM_EXTRACTION_TEMPLATE",
    "RELATION_INFERENCE_TEMPLATE",
    "ARGUMENT_MINING_TEMPLATE",
    "build_argument_mining_pipeline",
    "build_argument_miner",
    "build_argument_mining_hook",
    "ExtractedClaim",
    "HookedArgumentMiner",
    "LongDocumentMiner",
    "Segmenter",
    "SimpleArgumentMiner",
]
