"""Run synthetic-only QA checks on chat dataset JSONL files."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from arglib.io import (
    validate_chat_deferred_relevance_dict,
    validate_chat_exploration_dict,
    validate_chat_grounded_dict,
    validate_chat_grounded_evidence_dict,
)

Validator = Callable[[dict[str, Any]], list[str]]


def _resolve_validator(kind: str) -> Validator:
    if kind == "chat_grounded":
        return validate_chat_grounded_dict
    if kind == "chat_exploration":
        return validate_chat_exploration_dict
    if kind == "chat_deferred_relevance":
        return validate_chat_deferred_relevance_dict
    if kind == "chat_grounded_evidence":
        return validate_chat_grounded_evidence_dict
    raise ValueError(f"Unknown dataset kind: {kind}")


def check_dataset(path: Path, *, kind: str) -> dict[str, Any]:
    dataset_kind = kind
    validator = _resolve_validator(dataset_kind)
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    invalid_rows = 0
    validation_errors = 0
    sample_errors: list[dict[str, Any]] = []
    claims_total = 0
    relations_total = 0
    self_loops = 0
    contradictory_pairs = 0
    claims_without_turn = 0
    claims_on_non_assistant_turn = 0
    claims_with_source_span = 0
    scenario_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()

    for idx, row in enumerate(rows):
        errors = validator(row)
        if errors:
            invalid_rows += 1
            validation_errors += len(errors)
            if len(sample_errors) < 10:
                sample_errors.append({"row_index": idx, "errors": errors[:5]})

        scenario_counts[str(row.get("scenario", "unknown"))] += 1
        split_counts[str(row.get("split", "unknown"))] += 1

        conversation = row.get("conversation", [])
        turn_role = {
            str(turn.get("turn_id")): str(turn.get("role"))
            for turn in conversation
            if isinstance(turn, dict)
        }
        labels = row.get("labels", {})
        claims = labels.get("claims", []) if isinstance(labels, dict) else []
        relations = labels.get("relations", []) if isinstance(labels, dict) else []
        claims_total += len(claims)
        relations_total += len(relations)

        claim_ids: set[str] = set()
        for claim in claims:
            if not isinstance(claim, dict):
                continue
            claim_id = str(claim.get("id"))
            claim_ids.add(claim_id)
            turn_id = str(claim.get("turn_id", ""))
            if not turn_id or turn_id not in turn_role:
                claims_without_turn += 1
            elif turn_role[turn_id] != "assistant":
                claims_on_non_assistant_turn += 1
            if claim.get("source_span") is not None:
                claims_with_source_span += 1

        pair_kinds: dict[tuple[str, str], set[str]] = {}
        for relation in relations:
            if not isinstance(relation, dict):
                continue
            src = str(relation.get("src"))
            dst = str(relation.get("dst"))
            rel_kind = str(relation.get("kind"))
            if src == dst and src in claim_ids:
                self_loops += 1
            pair_kinds.setdefault((src, dst), set()).add(rel_kind)
        contradictory_pairs += sum(
            1
            for kinds in pair_kinds.values()
            if "support" in kinds and "attack" in kinds
        )

    claim_per_row = claims_total / len(rows) if rows else 0.0
    relation_per_row = relations_total / len(rows) if rows else 0.0
    source_span_coverage = (
        claims_with_source_span / claims_total if claims_total else 0.0
    )

    checks = {
        "no_invalid_rows": invalid_rows == 0,
        "no_claims_without_turn": claims_without_turn == 0,
        "assistant_turn_claims_only": claims_on_non_assistant_turn == 0,
        "avg_claims_per_row_min_2": claim_per_row >= 2.0,
        "avg_relations_per_row_min_1": relation_per_row >= 1.0,
        "self_loops_below_5pct_relations": (
            self_loops / relations_total <= 0.05 if relations_total else True
        ),
    }

    return {
        "dataset": str(path),
        "kind": dataset_kind,
        "row_count": len(rows),
        "invalid_rows": invalid_rows,
        "validation_error_count": validation_errors,
        "sample_errors": sample_errors,
        "stats": {
            "claim_count": claims_total,
            "relation_count": relations_total,
            "avg_claims_per_row": claim_per_row,
            "avg_relations_per_row": relation_per_row,
            "self_loops": self_loops,
            "contradictory_relation_pairs": contradictory_pairs,
            "claims_without_turn": claims_without_turn,
            "claims_on_non_assistant_turn": claims_on_non_assistant_turn,
            "source_span_coverage": source_span_coverage,
            "scenario_distribution": dict(scenario_counts),
            "split_distribution": dict(split_counts),
        },
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Path to dataset JSONL file.")
    parser.add_argument(
        "--kind",
        required=True,
        choices=[
            "chat_grounded",
            "chat_exploration",
            "chat_deferred_relevance",
            "chat_grounded_evidence",
        ],
        help="Dataset schema kind.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write report JSON.",
    )
    args = parser.parse_args()

    report = check_dataset(Path(args.dataset), kind=args.kind)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved report: {output}")


if __name__ == "__main__":
    main()
