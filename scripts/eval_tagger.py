"""Evaluate ClaimRelationTagger against labeled JSONL datasets."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from arglib.ai import ClaimRelationTagger
from arglib.ai.mining import token_jaccard_similarity

LABELS = ("fact", "value", "policy", "other")


@dataclass(frozen=True)
class Interval:
    start: int
    end: int

    @property
    def length(self) -> int:
        return max(0, self.end - self.start)


def _normalize_claim_type(value: str) -> str:
    lowered = value.strip().lower()
    if lowered in {"fact", "factual"}:
        return "fact"
    if lowered == "value":
        return "value"
    if lowered == "policy":
        return "policy"
    return "other"


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9']+", text.lower()))


def _find_span(text: str, needle: str, used_starts: set[int]) -> Interval | None:
    if not needle.strip():
        return None
    start = 0
    lowered = text.lower()
    target = needle.lower().strip()
    while True:
        idx = lowered.find(target, start)
        if idx < 0:
            return None
        if idx not in used_starts:
            used_starts.add(idx)
            return Interval(start=idx, end=idx + len(target))
        start = idx + 1


def _interval_overlap(left: Interval, right: Interval) -> int:
    return max(0, min(left.end, right.end) - max(left.start, right.start))


def _span_prf(tp: int, pred: int, gold: int) -> dict[str, float]:
    precision = tp / pred if pred else 0.0
    recall = tp / gold if gold else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def _claim_type_scores(pairs: list[tuple[str, str]]) -> dict[str, Any]:
    # pairs: (gold, pred)
    per_label: dict[str, dict[str, int]] = {
        label: {"tp": 0, "fp": 0, "fn": 0} for label in LABELS
    }
    correct = 0
    for gold, pred in pairs:
        if gold == pred:
            correct += 1
        for label in LABELS:
            if pred == label and gold == label:
                per_label[label]["tp"] += 1
            elif pred == label and gold != label:
                per_label[label]["fp"] += 1
            elif gold == label and pred != label:
                per_label[label]["fn"] += 1

    macro_f1_total = 0.0
    support = 0
    weighted_f1_total = 0.0
    per_label_out: dict[str, dict[str, float]] = {}
    for label in LABELS:
        tp = per_label[label]["tp"]
        fp = per_label[label]["fp"]
        fn = per_label[label]["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_label_out[label] = {"precision": precision, "recall": recall, "f1": f1}
        macro_f1_total += f1
        label_support = tp + fn
        support += label_support
        weighted_f1_total += f1 * label_support

    n = len(pairs)
    return {
        "accuracy": (correct / n) if n else 0.0,
        "macro_f1": macro_f1_total / len(LABELS),
        "weighted_f1": (weighted_f1_total / support) if support else 0.0,
        "support": n,
        "labels": per_label_out,
    }


def _align_claims(
    gold_nodes: list[dict[str, Any]],
    pred_units: list[dict[str, Any]],
    threshold: float = 0.35,
) -> list[tuple[int, int, float]]:
    candidates: list[tuple[float, int, int]] = []
    for gi, g in enumerate(gold_nodes):
        for pi, p in enumerate(pred_units):
            score = token_jaccard_similarity(str(g["text"]), str(p["text"]))
            if score >= threshold:
                candidates.append((score, gi, pi))
    candidates.sort(reverse=True)
    used_g: set[int] = set()
    used_p: set[int] = set()
    aligned: list[tuple[int, int, float]] = []
    for score, gi, pi in candidates:
        if gi in used_g or pi in used_p:
            continue
        aligned.append((gi, pi, score))
        used_g.add(gi)
        used_p.add(pi)
    return aligned


def evaluate(
    dataset_path: Path,
    *,
    max_examples: int | None = None,
) -> dict[str, Any]:
    tagger = ClaimRelationTagger()
    total_examples = 0

    span_exact_tp = 0
    span_exact_pred = 0
    span_exact_gold = 0
    token_overlap = 0
    token_pred_total = 0
    token_gold_total = 0

    claim_type_pairs: list[tuple[str, str]] = []

    edge_tp = 0
    edge_pred = 0
    edge_gold = 0
    support_tp = 0
    support_pred = 0
    support_gold = 0
    attack_tp = 0
    attack_pred = 0
    attack_gold = 0

    for line in dataset_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        essay = str(row.get("essay", ""))
        graph = row.get("graph", {})
        gold_nodes = list(graph.get("nodes", []))
        gold_edges = list(graph.get("edges", []))
        if not essay or not gold_nodes:
            continue

        total_examples += 1
        if max_examples is not None and total_examples > max_examples:
            break

        tagged = tagger.tag(essay, doc_id=f"eval-{total_examples}")
        pred_units = list(tagged.graph.units.values())
        pred_rels = list(tagged.graph.relations)

        # Span metrics.
        gold_spans: list[Interval] = []
        used_starts: set[int] = set()
        for node in gold_nodes:
            span = _find_span(essay, str(node.get("text", "")), used_starts)
            if span is not None:
                gold_spans.append(span)
        pred_spans: list[Interval] = []
        for unit in pred_units:
            for span in unit.spans:
                pred_spans.append(Interval(span.start, span.end))

        span_exact_pred += len(pred_spans)
        span_exact_gold += len(gold_spans)
        unmatched_gold = set(range(len(gold_spans)))
        for pred in pred_spans:
            matched_idx: int | None = None
            for gi in unmatched_gold:
                gold = gold_spans[gi]
                if pred.start == gold.start and pred.end == gold.end:
                    matched_idx = gi
                    break
            if matched_idx is not None:
                span_exact_tp += 1
                unmatched_gold.remove(matched_idx)

        for pred in pred_spans:
            token_pred_total += pred.length
            best = 0
            for gold in gold_spans:
                best = max(best, _interval_overlap(pred, gold))
            token_overlap += best
        for gold in gold_spans:
            token_gold_total += gold.length

        # Claim alignment for type + edge mapping.
        gold_nodes_norm = [
            {
                "id": int(node["id"]),
                "text": str(node.get("text", "")),
                "type": _normalize_claim_type(str(node.get("type", "other"))),
            }
            for node in gold_nodes
        ]
        pred_nodes_norm = [
            {"id": unit.id, "text": unit.text, "type": _normalize_claim_type(unit.type)}
            for unit in pred_units
        ]
        aligned = _align_claims(gold_nodes_norm, pred_nodes_norm)
        gold_to_pred: dict[int, str] = {}
        pred_to_gold: dict[str, int] = {}
        for gi, pi, _score in aligned:
            gold = gold_nodes_norm[gi]
            pred = pred_nodes_norm[pi]
            claim_type_pairs.append((gold["type"], pred["type"]))
            gold_to_pred[gold["id"]] = pred["id"]
            pred_to_gold[pred["id"]] = gold["id"]

        gold_edge_set: set[tuple[int, int, str]] = set()
        for edge in gold_edges:
            kind = str(edge.get("relation", "")).lower()
            if kind == "supports":
                kind = "support"
            if kind == "attacks":
                kind = "attack"
            if kind not in {"support", "attack"}:
                continue
            src = int(edge["from"])
            dst = int(edge["to"])
            gold_edge_set.add((src, dst, kind))

        pred_edge_set: set[tuple[int, int, str]] = set()
        for rel in pred_rels:
            if rel.kind not in {"support", "attack"}:
                continue
            src_gold = pred_to_gold.get(rel.src)
            dst_gold = pred_to_gold.get(rel.dst)
            if src_gold is None or dst_gold is None:
                continue
            pred_edge_set.add((src_gold, dst_gold, rel.kind))

        edge_pred += len(pred_edge_set)
        edge_gold += len(gold_edge_set)
        for edge in pred_edge_set:
            if edge in gold_edge_set:
                edge_tp += 1
            if edge[2] == "support":
                support_pred += 1
            else:
                attack_pred += 1
        for edge in gold_edge_set:
            if edge[2] == "support":
                support_gold += 1
            else:
                attack_gold += 1
        for edge in gold_edge_set & pred_edge_set:
            if edge[2] == "support":
                support_tp += 1
            else:
                attack_tp += 1

    span_exact = _span_prf(span_exact_tp, span_exact_pred, span_exact_gold)
    token_precision = token_overlap / token_pred_total if token_pred_total else 0.0
    token_recall = token_overlap / token_gold_total if token_gold_total else 0.0
    token_f1 = (
        2 * token_precision * token_recall / (token_precision + token_recall)
        if token_precision + token_recall
        else 0.0
    )
    type_scores = _claim_type_scores(claim_type_pairs)

    return {
        "dataset": str(dataset_path),
        "examples_evaluated": total_examples,
        "claim_span_exact": span_exact,
        "claim_span_token": {
            "precision": token_precision,
            "recall": token_recall,
            "f1": token_f1,
        },
        "claim_type": type_scores,
        "relation_edge": _span_prf(edge_tp, edge_pred, edge_gold),
        "relation_support": _span_prf(support_tp, support_pred, support_gold),
        "relation_attack": _span_prf(attack_tp, attack_pred, attack_gold),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default=".external/argument-mining/processed_data/augmented.jsonl",
        help="Path to labeled JSONL dataset",
    )
    parser.add_argument(
        "--output",
        default="reports/tagger_baseline_augmented.json",
        help="Where to write report JSON",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional cap for quick iteration.",
    )
    args = parser.parse_args()

    dataset = Path(args.dataset)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    report = evaluate(dataset, max_examples=args.max_examples)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved report: {output}")


if __name__ == "__main__":
    main()
