"""Transformer-backed neural classifiers for tagging tasks."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_IMPORT_ERROR: Exception | None = None
try:
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoModelForTokenClassification,
        AutoTokenizer,
    )

    _TRANSFORMERS_AVAILABLE = True
except Exception as exc:  # pragma: no cover - env dependent
    _TRANSFORMERS_AVAILABLE = False
    _IMPORT_ERROR = exc


def transformers_available() -> bool:
    return _TRANSFORMERS_AVAILABLE


def require_transformers() -> None:
    if _TRANSFORMERS_AVAILABLE:
        return
    detail = f"{type(_IMPORT_ERROR).__name__}: {_IMPORT_ERROR}" if _IMPORT_ERROR else ""
    raise RuntimeError(
        "Transformers backend is unavailable. Install optional deps "
        "(`transformers`, `torch`, and `accelerate`) to use neural tagging. "
        f"{detail}".strip()
    )


def _softmax(values: list[float]) -> list[float]:
    if not values:
        return []
    max_value = max(values)
    exps = [math.exp(v - max_value) for v in values]
    total = sum(exps) or 1.0
    return [v / total for v in exps]


@dataclass
class NeuralSequenceClassifier:
    model_dir: str
    labels: tuple[str, ...]
    tokenizer: Any
    model: Any

    @classmethod
    def load(cls, model_dir: str | Path) -> NeuralSequenceClassifier:
        require_transformers()
        model_path = Path(model_dir)
        labels_path = model_path / "labels.json"
        payload = json.loads(labels_path.read_text(encoding="utf-8"))
        labels = tuple(str(label) for label in payload["labels"])
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
        model.eval()
        return cls(
            model_dir=str(model_path),
            labels=labels,
            tokenizer=tokenizer,
            model=model,
        )

    def predict(
        self, text: str, text_pair: str | None = None
    ) -> tuple[str, dict[str, float]]:
        require_transformers()
        inputs = self.tokenizer(
            text,
            text_pair,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits[0].tolist()
        probs = _softmax([float(v) for v in logits])
        if not probs:
            return self.labels[0], {self.labels[0]: 1.0}
        best_idx = max(range(len(probs)), key=probs.__getitem__)
        best_label = self.labels[best_idx]
        return best_label, {label: probs[idx] for idx, label in enumerate(self.labels)}


@dataclass
class NeuralTokenClassifier:
    model_dir: str
    labels: tuple[str, ...]
    tokenizer: Any
    model: Any

    @classmethod
    def load(cls, model_dir: str | Path) -> NeuralTokenClassifier:
        require_transformers()
        model_path = Path(model_dir)
        labels_path = model_path / "labels.json"
        payload = json.loads(labels_path.read_text(encoding="utf-8"))
        labels = tuple(str(label) for label in payload["labels"])
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        model = AutoModelForTokenClassification.from_pretrained(str(model_path))
        model.eval()
        return cls(
            model_dir=str(model_path),
            labels=labels,
            tokenizer=tokenizer,
            model=model,
        )

    def predict(self, text: str) -> list[tuple[str, str, float]]:
        require_transformers()
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            return_offsets_mapping=True,
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        with torch.no_grad():
            logits = self.model(**encoded).logits[0]
        probs = torch.softmax(logits, dim=-1)
        label_ids = torch.argmax(probs, dim=-1).tolist()
        confs = torch.max(probs, dim=-1).values.tolist()
        input_ids = encoded["input_ids"][0].tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        out: list[tuple[str, str, float]] = []
        for token, label_id, conf, offset in zip(
            tokens, label_ids, confs, offsets, strict=False
        ):
            start, end = int(offset[0]), int(offset[1])
            if end <= start:
                continue
            label = (
                self.labels[label_id]
                if label_id < len(self.labels)
                else self.labels[0]
            )
            out.append((token, label, float(conf)))
        return out


@dataclass
class NeuralTaggerBundle:
    span_tagging: NeuralTokenClassifier | None = None
    claim_type: NeuralSequenceClassifier | None = None
    relation: NeuralSequenceClassifier | None = None
    coref_link: NeuralSequenceClassifier | None = None
    deferred_pending: NeuralSequenceClassifier | None = None
    deferred_resolution: NeuralSequenceClassifier | None = None
    discourse_function: NeuralSequenceClassifier | None = None
    contradiction_nli: NeuralSequenceClassifier | None = None
    evidence_stance: NeuralSequenceClassifier | None = None
    evidence_retrieval: NeuralSequenceClassifier | None = None

    @classmethod
    def load(cls, model_dir: str | Path) -> NeuralTaggerBundle:
        model_root = Path(model_dir)
        if not model_root.exists():
            raise FileNotFoundError(f"Neural model dir does not exist: {model_root}")
        parts: dict[str, NeuralSequenceClassifier | None] = {}
        token_tasks = {"span_tagging"}
        task_aliases = {
            "claim_type": "claim_type",
            "relation_class": "relation",
            "coref_link": "coref_link",
            "deferred_pending": "deferred_pending",
            "deferred_resolution": "deferred_resolution",
            "discourse_function": "discourse_function",
            "contradiction_nli": "contradiction_nli",
            "claim_evidence_stance": "evidence_stance",
            "evidence_retrieval": "evidence_retrieval",
            "span_tagging": "span_tagging",
        }
        for task, alias in task_aliases.items():
            task_dir = model_root / task
            if not task_dir.exists():
                parts[alias] = None
                continue
            if task in token_tasks:
                parts[alias] = NeuralTokenClassifier.load(task_dir)
            else:
                parts[alias] = NeuralSequenceClassifier.load(task_dir)
        return cls(
            span_tagging=parts["span_tagging"],
            claim_type=parts["claim_type"],
            relation=parts["relation"],
            coref_link=parts["coref_link"],
            deferred_pending=parts["deferred_pending"],
            deferred_resolution=parts["deferred_resolution"],
            discourse_function=parts["discourse_function"],
            contradiction_nli=parts["contradiction_nli"],
            evidence_stance=parts["evidence_stance"],
            evidence_retrieval=parts["evidence_retrieval"],
        )


__all__ = [
    "NeuralSequenceClassifier",
    "NeuralTokenClassifier",
    "NeuralTaggerBundle",
    "require_transformers",
    "transformers_available",
]
