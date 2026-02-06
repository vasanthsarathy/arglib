"""Fast claim and relation tagging with incremental conversation memory."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from arglib.ai.artifacts import resolve_or_download
from arglib.ai.mining import token_jaccard_similarity
from arglib.ai.small_model import (
    SmallTaggerBundle,
    evidence_features,
    relation_features,
    tokenize,
)
from arglib.core import ArgumentGraph, TextSpan

RelationKind = Literal["support", "attack", "undercut", "rebut"]
ClaimType = Literal["fact", "value", "policy", "other"]

_SENTENCE_PATTERN = re.compile(r"[^.!?\n]+[.!?]?")
_POLICY_HINTS = {
    "should",
    "must",
    "need",
    "needs",
    "ought",
    "recommend",
    "propose",
    "ban",
    "require",
    "mandate",
}
_VALUE_HINTS = {
    "good",
    "bad",
    "better",
    "worse",
    "best",
    "worst",
    "fair",
    "unfair",
    "ethical",
    "unethical",
    "harmful",
    "beneficial",
}
_NEGATION_TOKENS = {"not", "no", "never", "none", "cannot", "can't", "won't"}
_SUPPORT_PREFIXES = ("because", "since", "given that")
_CONCLUSION_PREFIXES = ("therefore", "thus", "so", "hence", "consequently")
_ATTACK_PREFIXES = ("however", "but", "although", "though", "yet", "nevertheless")


@dataclass(frozen=True)
class TaggedClaim:
    claim_id: str
    text: str
    claim_type: ClaimType
    start: int
    end: int
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaggedRelation:
    src: str
    dst: str
    kind: RelationKind
    confidence: float
    rationale: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvidenceLinkPrediction:
    claim_id: str
    evidence_id: str
    retrieval_label: str
    retrieval_confidence: float
    stance: str
    stance_confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaggingResult:
    graph: ArgumentGraph
    claims: list[TaggedClaim]
    relations: list[TaggedRelation]


@dataclass
class ClaimRelationTagger:
    """Fast deterministic tagger for claims, claim types, and relations."""

    min_chars: int = 6
    relation_similarity_threshold: float = 0.34
    contradiction_similarity_threshold: float = 0.55

    def tag(
        self,
        text: str,
        *,
        doc_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TaggingResult:
        graph = ArgumentGraph.new()
        if metadata:
            graph.metadata.update(metadata)
        graph.metadata.setdefault("tagger", "fast-deterministic")

        claims: list[TaggedClaim] = []
        source_id = doc_id or "document"
        for match in _SENTENCE_PATTERN.finditer(text):
            sentence = match.group(0).strip()
            if not sentence or len(sentence) < self.min_chars or sentence.endswith("?"):
                continue
            claim_type, confidence = _classify_claim_type(sentence)
            claim_id = f"c{len(claims) + 1}"
            span = TextSpan(
                doc_id=source_id,
                start=match.start(),
                end=match.end(),
                text=sentence,
                modality="text",
            )
            graph.add_claim(
                sentence,
                claim_id=claim_id,
                type=claim_type,
                spans=[span],
                metadata={"extraction": "fast", "confidence": confidence},
            )
            claims.append(
                TaggedClaim(
                    claim_id=claim_id,
                    text=sentence,
                    claim_type=claim_type,
                    start=match.start(),
                    end=match.end(),
                    confidence=confidence,
                )
            )

        relations = self._infer_relations(claims)
        for relation in relations:
            graph.add_relation(
                relation.src,
                relation.dst,
                kind=relation.kind,
                weight=relation.confidence,
                rationale=relation.rationale,
                metadata=dict(relation.metadata),
            )
        return TaggingResult(graph=graph, claims=claims, relations=relations)

    def _infer_relations(self, claims: list[TaggedClaim]) -> list[TaggedRelation]:
        if len(claims) < 2:
            return []
        relations: list[TaggedRelation] = []
        seen: set[tuple[str, str, RelationKind]] = set()

        for index in range(1, len(claims)):
            prev_claim = claims[index - 1]
            current_claim = claims[index]
            lowered = current_claim.text.lower().lstrip()
            kind: RelationKind | None = None
            rationale = ""
            confidence = 0.72
            src = current_claim.claim_id
            dst = prev_claim.claim_id

            if lowered.startswith(_SUPPORT_PREFIXES):
                kind = "support"
                rationale = "Discourse cue indicates a supporting premise."
            elif lowered.startswith(_CONCLUSION_PREFIXES):
                kind = "support"
                src = prev_claim.claim_id
                dst = current_claim.claim_id
                rationale = "Discourse cue indicates a stated conclusion."
            elif lowered.startswith(_ATTACK_PREFIXES):
                kind = "attack"
                rationale = "Contrastive cue indicates disagreement."

            if kind is not None:
                key = (src, dst, kind)
                if key not in seen:
                    relations.append(
                        TaggedRelation(
                            src=src,
                            dst=dst,
                            kind=kind,
                            confidence=confidence,
                            rationale=rationale,
                            metadata={"inferred": True, "mode": "discourse"},
                        )
                    )
                    seen.add(key)

        for src_claim in claims:
            for dst_claim in claims:
                if src_claim.claim_id == dst_claim.claim_id:
                    continue
                similarity = token_jaccard_similarity(src_claim.text, dst_claim.text)
                if similarity < self.relation_similarity_threshold:
                    continue
                negation_flip = _has_negation_flip(src_claim.text, dst_claim.text)
                kind = "attack" if negation_flip else "support"
                if (
                    kind == "attack"
                    and similarity < self.contradiction_similarity_threshold
                ):
                    continue
                key = (src_claim.claim_id, dst_claim.claim_id, kind)
                if key in seen:
                    continue
                relations.append(
                    TaggedRelation(
                        src=src_claim.claim_id,
                        dst=dst_claim.claim_id,
                        kind=kind,
                        confidence=round(similarity, 3),
                        rationale=(
                            "High lexical overlap with negation mismatch."
                            if negation_flip
                            else "Lexical overlap suggests related support."
                        ),
                        metadata={"inferred": True, "mode": "similarity"},
                    )
                )
                seen.add(key)
        return relations


@dataclass
class HybridClaimRelationTagger:
    """Model-backed tagger with deterministic fallback."""

    model_path: str | None = "models/small_tagger_v1.json"
    neural_model_dir: str | None = None
    auto_download_artifacts: bool = False
    artifact_cache_dir: str | None = None
    artifact_manifest_path: str | None = None
    prefer_neural: bool = True
    fallback: ClaimRelationTagger = field(default_factory=ClaimRelationTagger)
    min_claim_confidence: float = 0.45
    min_relation_confidence: float = 0.55
    relation_top_k_per_claim: int = 2
    _bundle: SmallTaggerBundle | None = field(init=False, default=None)
    _neural_bundle: Any | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        model_file = resolve_or_download(
            local_path=self.model_path,
            artifact_key="small_tagger_v1",
            auto_download=self.auto_download_artifacts,
            cache_dir=self.artifact_cache_dir,
            manifest_path=self.artifact_manifest_path,
        )
        if model_file is not None and model_file.is_file():
            self._bundle = SmallTaggerBundle.load(model_file)

        neural_dir = resolve_or_download(
            local_path=self.neural_model_dir,
            artifact_key="neural_tagger_v1",
            auto_download=self.auto_download_artifacts,
            cache_dir=self.artifact_cache_dir,
            manifest_path=self.artifact_manifest_path,
        )
        if neural_dir is not None:
            self._load_neural_bundle(str(neural_dir))

    def tag(
        self,
        text: str,
        *,
        doc_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TaggingResult:
        if self.prefer_neural and self._neural_bundle is not None:
            neural = self._tag_with_neural(text, doc_id=doc_id, metadata=metadata)
            if neural is not None:
                return neural
        if self._bundle is None:
            result = self.fallback.tag(text, doc_id=doc_id, metadata=metadata)
            result.graph.metadata.setdefault("tagger_mode", "fallback")
            return result

        claims = _extract_claim_candidates(text)
        if not claims:
            result = self.fallback.tag(text, doc_id=doc_id, metadata=metadata)
            result.graph.metadata.setdefault("tagger_mode", "fallback")
            return result

        graph = ArgumentGraph.new()
        if metadata:
            graph.metadata.update(metadata)
        graph.metadata["tagger"] = "hybrid-small-model"
        graph.metadata["tagger_mode"] = "hybrid"
        source_id = doc_id or "document"

        tagged_claims: list[TaggedClaim] = []
        for index, claim in enumerate(claims, start=1):
            pred_type, probs = self._bundle.claim.predict(tokenize(claim["text"]))
            confidence = float(probs.get(pred_type, 0.0))
            if confidence < self.min_claim_confidence:
                pred_type, confidence = _classify_claim_type(claim["text"])
            claim_id = f"c{index}"
            graph.add_claim(
                claim["text"],
                claim_id=claim_id,
                type=pred_type if pred_type in {"fact", "value", "policy"} else "other",
                spans=[
                    TextSpan(
                        doc_id=source_id,
                        start=claim["start"],
                        end=claim["end"],
                        text=claim["text"],
                        modality="text",
                    )
                ],
                metadata={"extraction": "hybrid", "confidence": confidence},
            )
            tagged_claims.append(
                TaggedClaim(
                    claim_id=claim_id,
                    text=claim["text"],
                    claim_type=pred_type
                    if pred_type in {"fact", "value", "policy"}
                    else "other",
                    start=claim["start"],
                    end=claim["end"],
                    confidence=confidence,
                )
            )

        relations = self._predict_relations(tagged_claims)
        if not relations:
            relations = self.fallback._infer_relations(tagged_claims)
        for rel in relations:
            graph.add_relation(
                rel.src,
                rel.dst,
                kind=rel.kind,
                weight=rel.confidence,
                rationale=rel.rationale,
                metadata=dict(rel.metadata),
            )
        return TaggingResult(graph=graph, claims=tagged_claims, relations=relations)

    def predict_evidence_links(
        self,
        *,
        claims: list[TaggedClaim],
        evidence_items: list[dict[str, str]],
        min_retrieval_confidence: float = 0.5,
    ) -> list[EvidenceLinkPrediction]:
        results: list[EvidenceLinkPrediction] = []
        for claim in claims:
            for item in evidence_items:
                evidence_id = str(item.get("id", ""))
                evidence_text = str(item.get("text", "")).strip()
                if not evidence_id or not evidence_text:
                    continue
                retrieval_label, retrieval_conf = self._predict_evidence_retrieval(
                    claim.text, evidence_text
                )
                if (
                    retrieval_label != "yes"
                    or retrieval_conf < min_retrieval_confidence
                ):
                    continue
                stance_label, stance_conf = self._predict_evidence_stance(
                    claim.text, evidence_text
                )
                results.append(
                    EvidenceLinkPrediction(
                        claim_id=claim.claim_id,
                        evidence_id=evidence_id,
                        retrieval_label=retrieval_label,
                        retrieval_confidence=retrieval_conf,
                        stance=stance_label,
                        stance_confidence=stance_conf,
                        metadata={"mode": "hybrid-evidence"},
                    )
                )
        return results

    def _predict_relations(self, claims: list[TaggedClaim]) -> list[TaggedRelation]:
        if self._bundle is None or len(claims) < 2:
            return []
        candidates: dict[str, list[tuple[float, TaggedRelation]]] = {}
        for src in claims:
            for dst in claims:
                if src.claim_id == dst.claim_id:
                    continue
                pred, probs = self._bundle.relation.predict(
                    relation_features(src.text, dst.text)
                )
                confidence = float(probs.get(pred, 0.0))
                if confidence < self.min_relation_confidence:
                    continue
                if pred not in {"support", "attack"}:
                    continue
                relation = TaggedRelation(
                    src=src.claim_id,
                    dst=dst.claim_id,
                    kind=pred,
                    confidence=confidence,
                    rationale="small-model inference",
                    metadata={"inferred": True, "mode": "small-model"},
                )
                candidates.setdefault(src.claim_id, []).append((confidence, relation))

        selected: list[TaggedRelation] = []
        seen: set[tuple[str, str, str]] = set()
        for _src_id, rels in candidates.items():
            rels.sort(key=lambda item: item[0], reverse=True)
            for _score, relation in rels[: self.relation_top_k_per_claim]:
                key = (relation.src, relation.dst, relation.kind)
                if key in seen:
                    continue
                selected.append(relation)
                seen.add(key)
        return selected

    def _tag_with_neural(
        self,
        text: str,
        *,
        doc_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TaggingResult | None:
        neural = self._neural_bundle
        if neural is None or neural.claim_type is None:
            return None
        candidates = _extract_claim_candidates(text)
        if not candidates:
            return None
        graph = ArgumentGraph.new()
        if metadata:
            graph.metadata.update(metadata)
        graph.metadata["tagger"] = "hybrid-neural-model"
        graph.metadata["tagger_mode"] = "neural"
        source_id = doc_id or "document"

        tagged_claims: list[TaggedClaim] = []
        for index, claim in enumerate(candidates, start=1):
            label, probs = neural.claim_type.predict(claim["text"])
            confidence = float(probs.get(label, 0.0))
            claim_type = label if label in {"fact", "value", "policy"} else "other"
            claim_id = f"c{index}"
            graph.add_claim(
                claim["text"],
                claim_id=claim_id,
                type=claim_type,
                spans=[
                    TextSpan(
                        doc_id=source_id,
                        start=claim["start"],
                        end=claim["end"],
                        text=claim["text"],
                        modality="text",
                    )
                ],
                metadata={"extraction": "neural", "confidence": confidence},
            )
            tagged_claims.append(
                TaggedClaim(
                    claim_id=claim_id,
                    text=claim["text"],
                    claim_type=claim_type,
                    start=claim["start"],
                    end=claim["end"],
                    confidence=confidence,
                )
            )
        relations = self._predict_relations_neural(tagged_claims)
        if not relations:
            relations = self.fallback._infer_relations(tagged_claims)
        for rel in relations:
            graph.add_relation(
                rel.src,
                rel.dst,
                kind=rel.kind,
                weight=rel.confidence,
                rationale=rel.rationale,
                metadata=dict(rel.metadata),
            )
        return TaggingResult(graph=graph, claims=tagged_claims, relations=relations)

    def _predict_relations_neural(
        self, claims: list[TaggedClaim]
    ) -> list[TaggedRelation]:
        neural = self._neural_bundle
        if neural is None or neural.relation is None or len(claims) < 2:
            return []
        candidates: dict[str, list[tuple[float, TaggedRelation]]] = {}
        for src in claims:
            for dst in claims:
                if src.claim_id == dst.claim_id:
                    continue
                pred, probs = neural.relation.predict(src.text, dst.text)
                confidence = float(probs.get(pred, 0.0))
                if confidence < self.min_relation_confidence:
                    continue
                if pred not in {"support", "attack"}:
                    continue
                relation = TaggedRelation(
                    src=src.claim_id,
                    dst=dst.claim_id,
                    kind=pred,
                    confidence=confidence,
                    rationale="neural-model inference",
                    metadata={"inferred": True, "mode": "neural-model"},
                )
                candidates.setdefault(src.claim_id, []).append((confidence, relation))
        selected: list[TaggedRelation] = []
        seen: set[tuple[str, str, str]] = set()
        for _src_id, rels in candidates.items():
            rels.sort(key=lambda item: item[0], reverse=True)
            for _score, relation in rels[: self.relation_top_k_per_claim]:
                key = (relation.src, relation.dst, relation.kind)
                if key in seen:
                    continue
                selected.append(relation)
                seen.add(key)
        return selected

    def _predict_evidence_retrieval(
        self, claim_text: str, evidence_text: str
    ) -> tuple[str, float]:
        neural = self._neural_bundle
        if neural is not None and neural.evidence_retrieval is not None:
            pred, probs = neural.evidence_retrieval.predict(claim_text, evidence_text)
            return pred, float(probs.get(pred, 0.0))
        bundle = self._bundle
        if bundle is not None and bundle.evidence_retrieval is not None:
            pred, probs = bundle.evidence_retrieval.predict(
                evidence_features(claim_text, evidence_text)
            )
            return pred, float(probs.get(pred, 0.0))
        lexical = token_jaccard_similarity(claim_text, evidence_text)
        label = "yes" if lexical >= 0.2 else "no"
        return label, lexical if label == "yes" else 1.0 - lexical

    def _predict_evidence_stance(
        self, claim_text: str, evidence_text: str
    ) -> tuple[str, float]:
        neural = self._neural_bundle
        if neural is not None and neural.evidence_stance is not None:
            pred, probs = neural.evidence_stance.predict(claim_text, evidence_text)
            return pred, float(probs.get(pred, 0.0))
        bundle = self._bundle
        if bundle is not None and bundle.evidence_stance is not None:
            pred, probs = bundle.evidence_stance.predict(
                evidence_features(claim_text, evidence_text)
            )
            return pred, float(probs.get(pred, 0.0))
        neg_flip = _has_negation_flip(claim_text, evidence_text)
        if neg_flip:
            return "attack", 0.6
        lexical = token_jaccard_similarity(claim_text, evidence_text)
        if lexical < 0.2:
            return "insufficient", 0.55
        return "support", min(0.85, max(0.55, lexical))

    def _load_neural_bundle(self, neural_model_dir: str) -> None:
        try:
            from arglib.ai.neural_model import NeuralTaggerBundle, require_transformers

            require_transformers()
            self._neural_bundle = NeuralTaggerBundle.load(neural_model_dir)
        except Exception:
            self._neural_bundle = None


@dataclass
class TurnUpdate:
    turn_id: str
    speaker: str
    added_claim_ids: list[str]
    merged_claim_ids: list[str]
    added_relations: list[dict[str, str]]
    contradictions: list[dict[str, Any]]
    pending_created: list[dict[str, Any]] = field(default_factory=list)
    pending_resolved: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ConversationMemory:
    """Incremental claim-memory graph for chat-style streams."""

    tagger: ClaimRelationTagger = field(default_factory=ClaimRelationTagger)
    merge_similarity_threshold: float = 0.9
    relation_similarity_threshold: float = 0.4
    graph: ArgumentGraph = field(default_factory=ArgumentGraph.new)
    _turn_count: int = 0

    def ingest_turn(
        self,
        text: str,
        *,
        speaker: str = "unknown",
        turn_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TurnUpdate:
        self._turn_count += 1
        local_turn_id = turn_id or f"turn-{self._turn_count}"
        tagged = self.tagger.tag(text, doc_id=local_turn_id, metadata=metadata)

        claim_id_map: dict[str, str] = {}
        added_claim_ids: list[str] = []
        merged_claim_ids: list[str] = []
        contradictions: list[dict[str, Any]] = []
        pending_created: list[dict[str, Any]] = []
        pending_resolved: list[dict[str, Any]] = []

        existing_before = set(self.graph.units.keys())
        for tagged_claim in tagged.claims:
            local_unit = tagged.graph.units[tagged_claim.claim_id]
            match = self._find_best_match(local_unit.text, exclude=set())
            if match and match[1] >= self.merge_similarity_threshold:
                existing_id = match[0]
                claim_id_map[tagged_claim.claim_id] = existing_id
                merged_claim_ids.append(existing_id)
                existing = self.graph.units[existing_id]
                existing.metadata.setdefault("seen_in_turns", []).append(local_turn_id)
                existing.metadata.setdefault("seen_by_speakers", []).append(speaker)
                for span in local_unit.spans:
                    if not any(
                        s.doc_id == span.doc_id
                        and s.start == span.start
                        and s.end == span.end
                        for s in existing.spans
                    ):
                        existing.spans.append(span)
                continue

            new_id = self.graph._next_id("c")
            self.graph.add_claim(
                local_unit.text,
                claim_id=new_id,
                type=local_unit.type,
                spans=list(local_unit.spans),
                metadata={
                    **dict(local_unit.metadata),
                    "turn_id": local_turn_id,
                    "speaker": speaker,
                },
            )
            claim_id_map[tagged_claim.claim_id] = new_id
            added_claim_ids.append(new_id)

        added_relations: list[dict[str, str]] = []
        for relation in tagged.graph.relations:
            src = claim_id_map.get(relation.src)
            dst = claim_id_map.get(relation.dst)
            if src is None or dst is None or src == dst:
                continue
            if self._has_relation(src, dst, relation.kind):
                continue
            self.graph.add_relation(
                src,
                dst,
                kind=relation.kind,
                weight=relation.weight,
                rationale=relation.rationale,
                metadata={
                    "turn_id": local_turn_id,
                    "speaker": speaker,
                    **relation.metadata,
                },
            )
            added_relations.append({"src": src, "dst": dst, "kind": relation.kind})

        for new_id in added_claim_ids:
            new_unit = self.graph.units[new_id]
            for existing_id in existing_before:
                existing_unit = self.graph.units[existing_id]
                similarity = token_jaccard_similarity(new_unit.text, existing_unit.text)
                if similarity < self.relation_similarity_threshold:
                    continue
                negation_flip = _has_negation_flip(new_unit.text, existing_unit.text)
                kind: RelationKind = "attack" if negation_flip else "support"
                if self._has_relation(new_id, existing_id, kind):
                    continue
                self.graph.add_relation(
                    new_id,
                    existing_id,
                    kind=kind,
                    weight=round(similarity, 3),
                    rationale=(
                        "Potential contradiction with prior memory."
                        if negation_flip
                        else "Semantically related to prior memory."
                    ),
                    metadata={
                        "turn_id": local_turn_id,
                        "speaker": speaker,
                        "memory_link": True,
                    },
                )
                added_relations.append(
                    {"src": new_id, "dst": existing_id, "kind": kind}
                )
                if kind == "attack":
                    contradictions.append(
                        {
                            "new_claim_id": new_id,
                            "prior_claim_id": existing_id,
                            "similarity": round(similarity, 3),
                            "reason": "negation_flip",
                        }
                    )
                    self.graph.units[existing_id].metadata["challenged_by"] = new_id

        if _contains_anaphoric_cue(text):
            for new_id in added_claim_ids:
                if any(
                    rel.src == new_id or rel.dst == new_id
                    for rel in self.graph.relations
                ):
                    continue
                recent_ids = self._recent_claim_ids(
                    exclude=set(added_claim_ids), limit=6
                )
                best_target: str | None = None
                best_score = -1.0
                for candidate_id in recent_ids:
                    candidate = self.graph.units.get(candidate_id)
                    source = self.graph.units.get(new_id)
                    if candidate is None or source is None:
                        continue
                    score = self._resolution_score(
                        pending_text=source.text, target_text=candidate.text
                    )
                    if score > best_score:
                        best_score = score
                        best_target = candidate_id
                if best_target is None:
                    continue
                source = self.graph.units.get(new_id)
                target = self.graph.units.get(best_target)
                if source is None or target is None:
                    continue
                kind = self._infer_link_kind(
                    source_text=source.text, target_text=target.text
                )
                if self._has_relation(new_id, best_target, kind):
                    continue
                self.graph.add_relation(
                    new_id,
                    best_target,
                    kind=kind,
                    weight=max(0.45, min(0.85, best_score)),
                    rationale="Anaphoric follow-up linked to recent memory.",
                    metadata={
                        "turn_id": local_turn_id,
                        "speaker": speaker,
                        "memory_link": True,
                        "anaphora_link": True,
                    },
                )
                added_relations.append(
                    {"src": new_id, "dst": best_target, "kind": kind}
                )

        pending_store = self._pending_store()
        if _contains_pending_cue(text):
            for claim_id in added_claim_ids:
                if any(item.get("claim_id") == claim_id for item in pending_store):
                    continue
                pending_item = {
                    "claim_id": claim_id,
                    "created_turn": local_turn_id,
                    "status": "pending",
                }
                pending_store.append(pending_item)
                self.graph.units[claim_id].metadata["pending_relevance"] = True
                pending_created.append(dict(pending_item))

        if _contains_resolution_cue(text):
            unresolved = [
                item for item in pending_store if item.get("status") == "pending"
            ]
            for item in unresolved:
                pending_id = str(item.get("claim_id"))
                target = self._find_resolution_target(
                    pending_id=pending_id,
                    candidate_ids=list(self.graph.units.keys()),
                )
                if target is None:
                    continue
                if not self._has_relation(pending_id, target, "support"):
                    self.graph.add_relation(
                        pending_id,
                        target,
                        kind="support",
                        weight=0.7,
                        rationale="Deferred relevance resolved by later explanation.",
                        metadata={
                            "turn_id": local_turn_id,
                            "speaker": speaker,
                            "deferred_resolution": True,
                        },
                    )
                    added_relations.append(
                        {"src": pending_id, "dst": target, "kind": "support"}
                    )
                item["status"] = "resolved"
                item["resolved_turn"] = local_turn_id
                self.graph.units[pending_id].metadata["pending_relevance"] = False
                self.graph.units[pending_id].metadata["resolved_relevance"] = {
                    "turn_id": local_turn_id,
                    "target_claim_id": target,
                }
                pending_resolved.append(
                    {
                        "pending_claim_id": pending_id,
                        "target_claim_id": target,
                        "resolver_turn": local_turn_id,
                    }
                )

        self.graph.metadata.setdefault("conversation", {})
        self.graph.metadata["conversation"].setdefault("turns", []).append(
            {
                "turn_id": local_turn_id,
                "speaker": speaker,
                "text": text,
                "claim_ids": list(added_claim_ids + merged_claim_ids),
            }
        )

        return TurnUpdate(
            turn_id=local_turn_id,
            speaker=speaker,
            added_claim_ids=added_claim_ids,
            merged_claim_ids=merged_claim_ids,
            added_relations=added_relations,
            contradictions=contradictions,
            pending_created=pending_created,
            pending_resolved=pending_resolved,
        )

    def _pending_store(self) -> list[dict[str, Any]]:
        self.graph.metadata.setdefault("conversation", {})
        return self.graph.metadata["conversation"].setdefault("pending_items", [])

    def _find_best_match(
        self, text: str, *, exclude: set[str]
    ) -> tuple[str, float] | None:
        best_id: str | None = None
        best_score = 0.0
        for unit_id, unit in self.graph.units.items():
            if unit_id in exclude:
                continue
            score = token_jaccard_similarity(text, unit.text)
            if score > best_score:
                best_score = score
                best_id = unit_id
        if best_id is None:
            return None
        return best_id, best_score

    def _has_relation(self, src: str, dst: str, kind: RelationKind) -> bool:
        return any(
            rel.src == src and rel.dst == dst and rel.kind == kind
            for rel in self.graph.relations
        )

    def _recent_claim_ids(self, *, exclude: set[str], limit: int = 6) -> list[str]:
        conversation = self.graph.metadata.get("conversation", {})
        turns = conversation.get("turns", [])
        if not isinstance(turns, list):
            return []
        ordered: list[str] = []
        for turn in reversed(turns):
            if not isinstance(turn, dict):
                continue
            claim_ids = turn.get("claim_ids", [])
            if not isinstance(claim_ids, list):
                continue
            for claim_id in reversed(claim_ids):
                cid = str(claim_id)
                if not cid or cid in exclude:
                    continue
                if cid not in self.graph.units:
                    continue
                if cid in ordered:
                    continue
                ordered.append(cid)
                if len(ordered) >= limit:
                    return ordered
        return ordered

    def _find_resolution_target(
        self, *, pending_id: str, candidate_ids: list[str]
    ) -> str | None:
        pending_unit = self.graph.units.get(pending_id)
        if pending_unit is None:
            return None
        best_id: str | None = None
        best_score = -1.0
        for candidate_id in candidate_ids:
            if candidate_id == pending_id:
                continue
            candidate = self.graph.units.get(candidate_id)
            if candidate is None:
                continue
            score = self._resolution_score(
                pending_text=pending_unit.text, target_text=candidate.text
            )
            if score > best_score:
                best_score = score
                best_id = candidate_id
        if best_id is None:
            for unit_id in self.graph.units:
                if unit_id != pending_id:
                    return unit_id
        return best_id

    def _resolution_score(self, *, pending_text: str, target_text: str) -> float:
        lexical = token_jaccard_similarity(pending_text, target_text)
        if isinstance(self.tagger, HybridClaimRelationTagger):
            bundle = self.tagger._bundle
            if bundle is not None:
                _pred, probs = bundle.relation.predict(
                    relation_features(pending_text, target_text)
                )
                support_prob = float(probs.get("support", 0.0))
                return 0.7 * support_prob + 0.3 * lexical
        return lexical

    def _infer_link_kind(self, *, source_text: str, target_text: str) -> RelationKind:
        if isinstance(self.tagger, HybridClaimRelationTagger):
            bundle = self.tagger._bundle
            if bundle is not None:
                pred, probs = bundle.relation.predict(
                    relation_features(source_text, target_text)
                )
                confidence = float(probs.get(pred, 0.0))
                if pred in {"support", "attack"} and confidence >= 0.52:
                    return pred
        src_pol = _polarity_score(source_text)
        dst_pol = _polarity_score(target_text)
        if src_pol < 0 and dst_pol < 0:
            return "support"
        if src_pol > 0 and dst_pol < 0:
            return "attack"
        if src_pol < 0 and dst_pol > 0:
            return "attack"
        return "support"


def _classify_claim_type(text: str) -> tuple[ClaimType, float]:
    tokens = set(re.findall(r"[a-zA-Z0-9']+", text.lower()))
    if tokens & _POLICY_HINTS:
        return "policy", 0.9
    if tokens & _VALUE_HINTS:
        return "value", 0.85
    if re.search(r"\b(is|are|was|were|has|have|will)\b", text.lower()):
        return "fact", 0.7
    return "other", 0.55


def _has_negation_flip(left: str, right: str) -> bool:
    left_tokens = set(re.findall(r"[a-zA-Z0-9']+", left.lower()))
    right_tokens = set(re.findall(r"[a-zA-Z0-9']+", right.lower()))
    left_neg = bool(left_tokens & _NEGATION_TOKENS)
    right_neg = bool(right_tokens & _NEGATION_TOKENS)
    return left_neg != right_neg


def _contains_pending_cue(text: str) -> bool:
    lowered = text.lower()
    cues = ("side note", "may seem unrelated", "for now", "set this aside")
    return any(cue in lowered for cue in cues)


def _contains_resolution_cue(text: str) -> bool:
    lowered = text.lower()
    cues = (
        "to clarify",
        "earlier side note",
        "the reason i inserted",
        "this is relevant because",
    )
    return any(cue in lowered for cue in cues)


def _contains_anaphoric_cue(text: str) -> bool:
    lowered = text.lower()
    cues = (
        "it ",
        "it's ",
        "this ",
        "that ",
        "these ",
        "those ",
        "i've used it",
        "i have used it",
    )
    return any(cue in lowered for cue in cues)


def _polarity_score(text: str) -> int:
    tokens = set(re.findall(r"[a-zA-Z0-9']+", text.lower()))
    positive = {
        "useful",
        "good",
        "great",
        "effective",
        "reliable",
        "helpful",
        "strong",
    }
    negative = {
        "bad",
        "terrible",
        "awful",
        "unreliable",
        "weak",
        "useless",
        "poor",
        "wrong",
    }
    return len(tokens & positive) - len(tokens & negative)


__all__ = [
    "ClaimRelationTagger",
    "HybridClaimRelationTagger",
    "ConversationMemory",
    "EvidenceLinkPrediction",
    "TaggedClaim",
    "TaggedRelation",
    "TaggingResult",
    "TurnUpdate",
]


def _extract_claim_candidates(text: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for match in _SENTENCE_PATTERN.finditer(text):
        sentence = match.group(0).strip()
        if not sentence or sentence.endswith("?"):
            continue
        start_base = match.start()
        for chunk in _split_sentence_chunks(sentence):
            chunk_text = chunk.strip(" ,;")
            if len(chunk_text) < 5:
                continue
            idx = sentence.find(chunk_text)
            if idx < 0:
                continue
            start = start_base + idx
            end = start + len(chunk_text)
            candidates.append({"text": chunk_text, "start": start, "end": end})
    return candidates


def _split_sentence_chunks(sentence: str) -> list[str]:
    lowered = sentence.lower()
    markers = [" because ", " therefore ", " however ", " but ", ";"]
    cuts: list[int] = []
    for marker in markers:
        start = 0
        while True:
            idx = lowered.find(marker, start)
            if idx < 0:
                break
            cuts.append(idx)
            start = idx + len(marker)
    if not cuts:
        return [sentence]
    cuts = sorted(set(cuts))
    chunks: list[str] = []
    prev = 0
    for cut in cuts:
        part = sentence[prev:cut].strip()
        if part:
            chunks.append(part)
        prev = cut
    tail = sentence[prev:].strip()
    if tail:
        chunks.append(tail)
    return chunks
