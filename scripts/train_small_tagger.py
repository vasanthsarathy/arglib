"""Train a tiny CPU-friendly model on synthetic chat corpora."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from arglib.ai.small_model import (
    CLAIM_LABELS,
    EVIDENCE_RETRIEVAL_LABELS,
    EVIDENCE_STANCE_LABELS,
    RELATION_LABELS,
    NaiveBayesTextClassifier,
    SmallTaggerBundle,
    evidence_features,
    relation_features,
    tokenize,
)


def _normalize_claim_type(value: str) -> str:
    lowered = value.lower().strip()
    if lowered in {"factual", "fact"}:
        return "fact"
    if lowered in {"value", "policy", "other"}:
        return lowered
    return "other"


def _macro_f1(labels: tuple[str, ...], pairs: list[tuple[str, str]]) -> float:
    if not pairs:
        return 0.0
    total = 0.0
    for label in labels:
        tp = sum(1 for g, p in pairs if g == label and p == label)
        fp = sum(1 for g, p in pairs if g != label and p == label)
        fn = sum(1 for g, p in pairs if g == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        total += f1
    return total / len(labels)


def _accuracy(pairs: list[tuple[str, str]]) -> float:
    if not pairs:
        return 0.0
    return sum(1 for gold, pred in pairs if gold == pred) / len(pairs)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _build_training_examples(rows: list[dict[str, Any]]) -> dict[str, Any]:
    claim_examples: dict[str, list[tuple[list[str], str]]] = defaultdict(list)
    relation_examples: dict[str, list[tuple[list[str], str]]] = defaultdict(list)
    evidence_stance_examples: dict[str, list[tuple[list[str], str]]] = defaultdict(list)
    evidence_retrieval_examples: dict[str, list[tuple[list[str], str]]] = defaultdict(
        list
    )

    for row in rows:
        split = str(row.get("split", "train"))
        labels = row.get("labels", {})
        claims = labels.get("claims", [])
        relations = labels.get("relations", [])
        evidence_items = labels.get("evidence_items", [])
        claim_evidence_links = labels.get("claim_evidence_links", [])
        claim_lookup: dict[str, str] = {}
        evidence_lookup: dict[str, str] = {}
        for claim in claims:
            if not isinstance(claim, dict):
                continue
            text = str(claim.get("text", "")).strip()
            if not text:
                continue
            ctype = _normalize_claim_type(str(claim.get("type", "other")))
            claim_id = str(claim.get("id"))
            claim_lookup[claim_id] = text
            claim_examples[split].append((tokenize(text), ctype))
        for evidence in evidence_items:
            if not isinstance(evidence, dict):
                continue
            evidence_id = str(evidence.get("id"))
            text = str(evidence.get("text", "")).strip()
            if not evidence_id or not text:
                continue
            evidence_lookup[evidence_id] = text
        for relation in relations:
            if not isinstance(relation, dict):
                continue
            src = claim_lookup.get(str(relation.get("src")))
            dst = claim_lookup.get(str(relation.get("dst")))
            kind = str(relation.get("kind", "support")).lower()
            if src is None or dst is None or kind not in RELATION_LABELS:
                continue
            relation_examples[split].append((relation_features(src, dst), kind))
        positive_pairs: set[tuple[str, str]] = set()
        for link in claim_evidence_links:
            if not isinstance(link, dict):
                continue
            claim_id = str(link.get("claim_id"))
            evidence_id = str(link.get("evidence_id"))
            stance = str(link.get("stance", "insufficient")).lower()
            claim_text = claim_lookup.get(claim_id)
            evidence_text = evidence_lookup.get(evidence_id)
            if claim_text is None or evidence_text is None:
                continue
            if stance not in EVIDENCE_STANCE_LABELS:
                stance = "insufficient"
            feats = evidence_features(claim_text, evidence_text)
            evidence_stance_examples[split].append((feats, stance))
            evidence_retrieval_examples[split].append((feats, "yes"))
            positive_pairs.add((claim_id, evidence_id))
        for claim_id, claim_text in claim_lookup.items():
            for evidence_id, evidence_text in evidence_lookup.items():
                if (claim_id, evidence_id) in positive_pairs:
                    continue
                feats = evidence_features(claim_text, evidence_text)
                evidence_retrieval_examples[split].append((feats, "no"))
    return {
        "claim": claim_examples,
        "relation": relation_examples,
        "evidence_stance": evidence_stance_examples,
        "evidence_retrieval": evidence_retrieval_examples,
    }


def train_and_evaluate(paths: list[Path]) -> dict[str, Any]:
    all_rows: list[dict[str, Any]] = []
    for path in paths:
        all_rows.extend(_load_rows(path))
    examples = _build_training_examples(all_rows)

    claim_train = examples["claim"].get("train", [])
    rel_train = examples["relation"].get("train", [])
    claim_model = NaiveBayesTextClassifier(labels=CLAIM_LABELS)
    relation_model = NaiveBayesTextClassifier(labels=RELATION_LABELS)
    evidence_stance_model = NaiveBayesTextClassifier(labels=EVIDENCE_STANCE_LABELS)
    evidence_retrieval_model = NaiveBayesTextClassifier(
        labels=EVIDENCE_RETRIEVAL_LABELS
    )
    claim_model.fit(claim_train)
    relation_model.fit(rel_train)
    evidence_stance_train = examples["evidence_stance"].get("train", [])
    evidence_retrieval_train = examples["evidence_retrieval"].get("train", [])
    evidence_stance_model.fit(evidence_stance_train)
    evidence_retrieval_model.fit(evidence_retrieval_train)

    bundle = SmallTaggerBundle(
        claim=claim_model,
        relation=relation_model,
        evidence_stance=evidence_stance_model,
        evidence_retrieval=evidence_retrieval_model,
    )
    out: dict[str, Any] = {
        "datasets": [str(p) for p in paths],
        "train_sizes": {
            "claim": len(claim_train),
            "relation": len(rel_train),
            "evidence_stance": len(evidence_stance_train),
            "evidence_retrieval": len(evidence_retrieval_train),
        },
        "splits": {},
        "model": bundle.to_dict(),
    }

    for split in ("train", "dev", "test"):
        claim_pairs: list[tuple[str, str]] = []
        for tokens, gold in examples["claim"].get(split, []):
            pred, _probs = claim_model.predict(tokens)
            claim_pairs.append((gold, pred))
        rel_pairs: list[tuple[str, str]] = []
        for feats, gold in examples["relation"].get(split, []):
            pred, _probs = relation_model.predict(feats)
            rel_pairs.append((gold, pred))
        out["splits"][split] = {
            "claim": {
                "n": len(claim_pairs),
                "accuracy": _accuracy(claim_pairs),
                "macro_f1": _macro_f1(CLAIM_LABELS, claim_pairs),
            },
            "relation": {
                "n": len(rel_pairs),
                "accuracy": _accuracy(rel_pairs),
                "macro_f1": _macro_f1(RELATION_LABELS, rel_pairs),
            },
            "evidence_stance": _eval_head(
                examples["evidence_stance"].get(split, []),
                model=evidence_stance_model,
                labels=EVIDENCE_STANCE_LABELS,
            ),
            "evidence_retrieval": _eval_head(
                examples["evidence_retrieval"].get(split, []),
                model=evidence_retrieval_model,
                labels=EVIDENCE_RETRIEVAL_LABELS,
            ),
        }
    return out


def _eval_head(
    examples: list[tuple[list[str], str]],
    *,
    model: NaiveBayesTextClassifier,
    labels: tuple[str, ...],
) -> dict[str, float | int]:
    pairs: list[tuple[str, str]] = []
    for feats, gold in examples:
        pred, _probs = model.predict(feats)
        pairs.append((gold, pred))
    return {
        "n": len(pairs),
        "accuracy": _accuracy(pairs),
        "macro_f1": _macro_f1(labels, pairs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "data/arg_mining/chat_grounded.jsonl",
            "data/arg_mining/chat_exploration.jsonl",
            "data/arg_mining/chat_deferred_relevance.jsonl",
            "data/arg_mining/chat_grounded_evidence.jsonl",
        ],
        help="Input JSONL dataset paths.",
    )
    parser.add_argument(
        "--model-output",
        default="models/small_tagger_v1.json",
        help="Path to write trained model JSON.",
    )
    parser.add_argument(
        "--report-output",
        default="reports/small_tagger_training_report.json",
        help="Path to write training report JSON.",
    )
    args = parser.parse_args()

    datasets = [Path(p) for p in args.datasets]
    report = train_and_evaluate(datasets)

    model_output = Path(args.model_output)
    report_output = Path(args.report_output)
    model_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    model_output.write_text(json.dumps(report["model"]), encoding="utf-8")
    report_output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    brief = {
        "train_sizes": report["train_sizes"],
        "dev": report["splits"].get("dev", {}),
        "test": report["splits"].get("test", {}),
    }
    print(json.dumps(brief, indent=2))
    print(f"\nSaved model: {model_output}")
    print(f"Saved report: {report_output}")


if __name__ == "__main__":
    main()
