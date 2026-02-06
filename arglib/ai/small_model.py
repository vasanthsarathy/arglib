"""Tiny CPU-friendly text classifiers used by hybrid tagging."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CLAIM_LABELS = ("fact", "value", "policy", "other")
RELATION_LABELS = ("support", "attack")
EVIDENCE_STANCE_LABELS = ("support", "attack", "insufficient")
EVIDENCE_RETRIEVAL_LABELS = ("yes", "no")


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9']+", text.lower())


def relation_features(src: str, dst: str) -> list[str]:
    src_tokens = tokenize(src)
    dst_tokens = tokenize(dst)
    src_set = set(src_tokens)
    dst_set = set(dst_tokens)
    overlap = len(src_set & dst_set)
    union = len(src_set | dst_set) or 1
    jaccard = overlap / union
    features: list[str] = []
    features.extend(f"src:{tok}" for tok in src_tokens)
    features.extend(f"dst:{tok}" for tok in dst_tokens)
    features.append(f"jaccard_bin:{int(jaccard * 10)}")
    src_neg = int(any(tok in {"not", "never", "no", "cannot"} for tok in src_tokens))
    dst_neg = int(any(tok in {"not", "never", "no", "cannot"} for tok in dst_tokens))
    features.append(f"src_neg:{src_neg}")
    features.append(f"dst_neg:{dst_neg}")
    features.append(f"neg_flip:{int(src_neg != dst_neg)}")
    return features


def evidence_features(claim_text: str, evidence_text: str) -> list[str]:
    features = relation_features(claim_text, evidence_text)
    claim_tokens = tokenize(claim_text)
    evidence_tokens = tokenize(evidence_text)
    claim_set = set(claim_tokens)
    evidence_set = set(evidence_tokens)
    overlap = len(claim_set & evidence_set)
    claim_len = max(1, len(claim_set))
    evidence_len = max(1, len(evidence_set))
    features.append(f"overlap_claim_ratio:{int((overlap / claim_len) * 10)}")
    features.append(f"overlap_evidence_ratio:{int((overlap / evidence_len) * 10)}")
    return features


def _softmax(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    max_score = max(scores.values())
    exps = {label: math.exp(value - max_score) for label, value in scores.items()}
    total = sum(exps.values()) or 1.0
    return {label: value / total for label, value in exps.items()}


@dataclass
class NaiveBayesTextClassifier:
    labels: tuple[str, ...]
    alpha: float = 1.0
    priors: dict[str, float] | None = None
    token_log_probs: dict[str, dict[str, float]] | None = None
    unk_log_prob: dict[str, float] | None = None

    def fit(self, examples: list[tuple[list[str], str]]) -> None:
        doc_counts: Counter[str] = Counter()
        token_counts: dict[str, Counter[str]] = {
            label: Counter() for label in self.labels
        }
        total_tokens: Counter[str] = Counter()
        vocab: set[str] = set()
        for tokens, label in examples:
            if label not in self.labels:
                continue
            doc_counts[label] += 1
            token_counts[label].update(tokens)
            total_tokens[label] += len(tokens)
            vocab.update(tokens)

        total_docs = sum(doc_counts.values())
        self.priors = {}
        self.token_log_probs = {}
        self.unk_log_prob = {}
        vocab_size = max(1, len(vocab))
        for label in self.labels:
            prior_num = doc_counts[label] + self.alpha
            prior_den = total_docs + self.alpha * len(self.labels)
            self.priors[label] = math.log(prior_num / prior_den) if prior_den else 0.0
            denom = total_tokens[label] + self.alpha * vocab_size
            self.token_log_probs[label] = {}
            for token in vocab:
                count = token_counts[label][token]
                prob = (count + self.alpha) / denom
                self.token_log_probs[label][token] = math.log(prob)
            self.unk_log_prob[label] = math.log(self.alpha / denom)

    def predict(self, tokens: list[str]) -> tuple[str, dict[str, float]]:
        if (
            self.priors is None
            or self.token_log_probs is None
            or self.unk_log_prob is None
        ):
            raise ValueError("Model is not initialized.")
        scores: dict[str, float] = {}
        for label in self.labels:
            score = self.priors[label]
            table = self.token_log_probs[label]
            unk = self.unk_log_prob[label]
            for token in tokens:
                score += table.get(token, unk)
            scores[label] = score
        best = max(scores.items(), key=lambda item: item[1])[0]
        return best, _softmax(scores)

    def to_dict(self) -> dict[str, Any]:
        return {
            "labels": list(self.labels),
            "alpha": self.alpha,
            "priors": self.priors or {},
            "token_log_probs": self.token_log_probs or {},
            "unk_log_prob": self.unk_log_prob or {},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> NaiveBayesTextClassifier:
        labels_raw = payload.get("labels", [])
        labels = tuple(str(label) for label in labels_raw)
        model = cls(labels=labels, alpha=float(payload.get("alpha", 1.0)))
        model.priors = {
            str(label): float(value)
            for label, value in payload.get("priors", {}).items()
        }
        model.token_log_probs = {
            str(label): {str(tok): float(v) for tok, v in table.items()}
            for label, table in payload.get("token_log_probs", {}).items()
        }
        model.unk_log_prob = {
            str(label): float(value)
            for label, value in payload.get("unk_log_prob", {}).items()
        }
        return model


@dataclass
class SmallTaggerBundle:
    claim: NaiveBayesTextClassifier
    relation: NaiveBayesTextClassifier
    evidence_stance: NaiveBayesTextClassifier | None = None
    evidence_retrieval: NaiveBayesTextClassifier | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {"claim": self.claim.to_dict(), "relation": self.relation.to_dict()}
        if self.evidence_stance is not None:
            payload["evidence_stance"] = self.evidence_stance.to_dict()
        if self.evidence_retrieval is not None:
            payload["evidence_retrieval"] = self.evidence_retrieval.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SmallTaggerBundle:
        evidence_stance_raw = payload.get("evidence_stance")
        evidence_retrieval_raw = payload.get("evidence_retrieval")
        return cls(
            claim=NaiveBayesTextClassifier.from_dict(dict(payload.get("claim", {}))),
            relation=NaiveBayesTextClassifier.from_dict(
                dict(payload.get("relation", {}))
            ),
            evidence_stance=(
                NaiveBayesTextClassifier.from_dict(dict(evidence_stance_raw))
                if isinstance(evidence_stance_raw, dict)
                else None
            ),
            evidence_retrieval=(
                NaiveBayesTextClassifier.from_dict(dict(evidence_retrieval_raw))
                if isinstance(evidence_retrieval_raw, dict)
                else None
            ),
        )

    @classmethod
    def load(cls, path: str | Path) -> SmallTaggerBundle:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict()), encoding="utf-8")


__all__ = [
    "CLAIM_LABELS",
    "EVIDENCE_RETRIEVAL_LABELS",
    "EVIDENCE_STANCE_LABELS",
    "RELATION_LABELS",
    "NaiveBayesTextClassifier",
    "SmallTaggerBundle",
    "evidence_features",
    "relation_features",
    "tokenize",
]
