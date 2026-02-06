"""Evaluate deferred relevance using runtime ConversationMemory behavior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from arglib.ai import ConversationMemory, HybridClaimRelationTagger
from arglib.ai.mining import token_jaccard_similarity


def _f1(tp: int, pred: int, gold: int) -> dict[str, float]:
    precision = tp / pred if pred else 0.0
    recall = tp / gold if gold else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def _align_gold_to_pred(
    gold_claims: list[dict[str, Any]],
    pred_units: dict[str, Any],
    *,
    threshold: float = 0.35,
) -> tuple[dict[str, str], dict[str, str]]:
    candidates: list[tuple[float, str, str]] = []
    for claim in gold_claims:
        gold_id = str(claim.get("id"))
        gold_text = str(claim.get("text", ""))
        for pred_id, unit in pred_units.items():
            score = token_jaccard_similarity(gold_text, unit.text)
            if score >= threshold:
                candidates.append((score, gold_id, pred_id))
    candidates.sort(reverse=True)
    gold_to_pred: dict[str, str] = {}
    pred_to_gold: dict[str, str] = {}
    used_gold: set[str] = set()
    used_pred: set[str] = set()
    for _score, gold_id, pred_id in candidates:
        if gold_id in used_gold or pred_id in used_pred:
            continue
        gold_to_pred[gold_id] = pred_id
        pred_to_gold[pred_id] = gold_id
        used_gold.add(gold_id)
        used_pred.add(pred_id)
    return gold_to_pred, pred_to_gold


def evaluate(
    path: Path, *, model_path: str | None = "models/small_tagger_v1.json"
) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    pending_tp = 0
    pending_pred_total = 0
    pending_gold_total = 0
    link_tp = 0
    link_pred_total = 0
    link_gold_total = 0
    promotion_tp = 0
    promotion_pred_total = 0
    promotion_gold_total = 0
    false_early_links = 0
    latency_errors: list[int] = []

    for row in rows:
        conversation = row.get("conversation", [])
        labels = row.get("labels", {})
        gold_claims = [c for c in labels.get("claims", []) if isinstance(c, dict)]
        deferred = labels.get("deferred_relevance", {})
        pending_items = deferred.get("pending_items", [])
        resolution_links = deferred.get("resolution_links", [])

        tagger = HybridClaimRelationTagger(model_path=model_path)
        memory = ConversationMemory(tagger=tagger)

        pred_pending_raw: set[str] = set()
        pred_links_raw: set[tuple[str, str, str]] = set()
        pred_created_turn_by_claim: dict[str, str] = {}
        for turn in conversation:
            if not isinstance(turn, dict):
                continue
            turn_id = str(turn.get("turn_id", ""))
            role = str(turn.get("role", "assistant"))
            text = str(turn.get("text", ""))
            if role not in {"assistant", "user", "system"}:
                role = "assistant"
            update = memory.ingest_turn(text, speaker=role, turn_id=turn_id)
            for item in update.pending_created:
                claim_id = str(item.get("claim_id", ""))
                if not claim_id:
                    continue
                pred_pending_raw.add(claim_id)
                pred_created_turn_by_claim[claim_id] = str(
                    item.get("created_turn", turn_id)
                )
            for item in update.pending_resolved:
                pending_claim_id = str(item.get("pending_claim_id", ""))
                target_claim_id = str(item.get("target_claim_id", ""))
                resolver_turn = str(item.get("resolver_turn", turn_id))
                if not pending_claim_id or not target_claim_id:
                    continue
                pred_links_raw.add((pending_claim_id, target_claim_id, resolver_turn))

        _, pred_to_gold = _align_gold_to_pred(gold_claims, memory.graph.units)

        gold_pending = {str(item.get("claim_id")) for item in pending_items}
        pending_gold_total += len(gold_pending)
        gold_links = {
            (
                str(item.get("pending_claim_id")),
                str(item.get("target_claim_id")),
                str(item.get("resolver_turn")),
            )
            for item in resolution_links
        }
        link_gold_total += len(gold_links)
        gold_promoted = {
            str(item.get("claim_id"))
            for item in pending_items
            if str(item.get("status")) == "resolved"
        }
        promotion_gold_total += len(gold_promoted)

        pred_pending = {
            pred_to_gold[pred_id]
            for pred_id in pred_pending_raw
            if pred_id in pred_to_gold
        }
        pred_links = {
            (pred_to_gold[src], pred_to_gold[dst], resolver_turn)
            for src, dst, resolver_turn in pred_links_raw
            if src in pred_to_gold and dst in pred_to_gold
        }

        pending_pred_total += len(pred_pending)
        pending_tp += len(pred_pending & gold_pending)
        link_pred_total += len(pred_links)
        link_tp += len(pred_links & gold_links)

        pred_promoted = {pending_id for pending_id, _, _ in pred_links}
        promotion_pred_total += len(pred_promoted)
        promotion_tp += len(pred_promoted & gold_promoted)

        gold_created_turn = {
            str(item.get("claim_id")): str(item.get("created_turn", "t0"))
            for item in pending_items
        }
        for pending_id, _target_id, resolver_turn in pred_links:
            created_turn = gold_created_turn.get(pending_id, "t0")
            if created_turn.startswith("t") and resolver_turn.startswith("t"):
                if int(resolver_turn[1:]) <= int(created_turn[1:]):
                    false_early_links += 1

        gold_latency = {}
        for item in pending_items:
            if str(item.get("status")) != "resolved":
                continue
            claim_id = str(item.get("claim_id"))
            created_turn = str(item.get("created_turn", "t0"))
            resolved_turn = str(item.get("resolved_turn", "t0"))
            if created_turn.startswith("t") and resolved_turn.startswith("t"):
                gold_latency[claim_id] = int(resolved_turn[1:]) - int(created_turn[1:])

        pred_latency = {}
        for pending_id, _, resolver_turn in pred_links:
            created_turn = gold_created_turn.get(pending_id, "t0")
            if created_turn.startswith("t") and resolver_turn.startswith("t"):
                pred_latency[pending_id] = int(resolver_turn[1:]) - int(
                    created_turn[1:]
                )

        for claim_id in set(gold_latency) & set(pred_latency):
            latency_errors.append(abs(pred_latency[claim_id] - gold_latency[claim_id]))

    latency_mae = sum(latency_errors) / len(latency_errors) if latency_errors else 0.0
    return {
        "dataset": str(path),
        "rows": len(rows),
        "mode": "runtime-memory",
        "pending_detection": _f1(pending_tp, pending_pred_total, pending_gold_total),
        "delayed_link_accuracy": _f1(link_tp, link_pred_total, link_gold_total),
        "promotion_detection": _f1(
            promotion_tp, promotion_pred_total, promotion_gold_total
        ),
        "false_early_link_rate": (
            false_early_links / link_pred_total if link_pred_total else 0.0
        ),
        "resolution_latency_mae": latency_mae,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="data/arg_mining/chat_deferred_relevance.jsonl",
    )
    parser.add_argument(
        "--output",
        default="reports/deferred_relevance_eval.json",
    )
    parser.add_argument(
        "--model-path",
        default="models/small_tagger_v1.json",
        help="Hybrid model path. Falls back to deterministic mode if missing.",
    )
    args = parser.parse_args()
    report = evaluate(Path(args.dataset), model_path=args.model_path)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved report: {output}")


if __name__ == "__main__":
    main()
