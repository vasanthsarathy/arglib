"""Build synthetic chat dataset with deferred relevance annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from arglib.io import validate_chat_deferred_relevance_payload


def _normalize_claim_type(value: str) -> str:
    lowered = value.strip().lower()
    if lowered in {"factual", "fact"}:
        return "fact"
    if lowered in {"value", "policy"}:
        return lowered
    return "other"


def _normalize_relation(value: str) -> str:
    lowered = value.strip().lower()
    if lowered in {"attack", "attacks"}:
        return "attack"
    return "support"


def _split_for_index(index: int) -> str:
    bucket = index % 10
    if bucket < 7:
        return "train"
    if bucket < 9:
        return "dev"
    return "test"


def _build_row(row: dict[str, Any], *, idx: int) -> dict[str, Any]:
    topic = str(row.get("topic", "Unknown topic")).strip() or "Unknown topic"
    graph = row.get("graph", {})
    nodes = list(graph.get("nodes", []))
    edges = list(graph.get("edges", []))
    if len(nodes) < 3:
        raise ValueError("Need at least 3 nodes for deferred relevance generation.")

    pending = nodes[0]
    anchor = nodes[1]
    resolver = nodes[2]

    c1 = {
        "id": "c1",
        "text": str(pending.get("text", "")).strip(),
        "type": _normalize_claim_type(str(pending.get("type", "other"))),
        "implicit": bool(pending.get("implicit", False)),
        "turn_id": "t2",
        "source_span": None,
    }
    c2 = {
        "id": "c2",
        "text": str(anchor.get("text", "")).strip(),
        "type": _normalize_claim_type(str(anchor.get("type", "other"))),
        "implicit": bool(anchor.get("implicit", False)),
        "turn_id": "t2",
        "source_span": None,
    }
    c3 = {
        "id": "c3",
        "text": str(resolver.get("text", "")).strip(),
        "type": _normalize_claim_type(str(resolver.get("type", "other"))),
        "implicit": bool(resolver.get("implicit", False)),
        "turn_id": "t4",
        "source_span": None,
    }

    relation_kind = "support"
    for edge in edges:
        if int(edge.get("from", -1)) == int(pending.get("id", -2)) and int(
            edge.get("to", -1)
        ) == int(anchor.get("id", -3)):
            relation_kind = _normalize_relation(str(edge.get("relation", "support")))
            break

    t2_text = (
        f"{c2['text']} "
        f"Side note (may seem unrelated for now): {c1['text']}"
    )
    t4_text = (
        f"To clarify why I inserted that earlier side note, {c1['text']} "
        f"This matters because it helps justify the main point: {c2['text']} "
        f"Additional explanation: {c3['text']}"
    )

    return {
        "id": f"cd-{idx:05d}",
        "split": _split_for_index(idx),
        "scenario": "research",
        "source_documents": [],
        "conversation": [
            {
                "turn_id": "t1",
                "role": "user",
                "text": f"We are exploring arguments about {topic}.",
                "discourse_function": "question",
            },
            {
                "turn_id": "t2",
                "role": "assistant",
                "text": t2_text,
                "discourse_function": "claim",
            },
            {
                "turn_id": "t3",
                "role": "user",
                "text": "The side note seems unrelated. Why include it?",
                "discourse_function": "question",
            },
            {
                "turn_id": "t4",
                "role": "assistant",
                "text": t4_text,
                "discourse_function": "explanation",
            },
        ],
        "labels": {
            "claims": [c1, c2, c3],
            "relations": [
                {"src": "c1", "dst": "c2", "kind": relation_kind},
                {"src": "c3", "dst": "c2", "kind": "support"},
            ],
            "deferred_relevance": {
                "pending_items": [
                    {
                        "claim_id": "c1",
                        "created_turn": "t2",
                        "status": "resolved",
                        "resolved_turn": "t4",
                    }
                ],
                "resolution_links": [
                    {
                        "pending_claim_id": "c1",
                        "resolver_turn": "t4",
                        "target_claim_id": "c2",
                        "kind": "context_support",
                        "confidence": 0.9,
                    }
                ],
            },
        },
    }


def build_dataset(
    input_path: Path, output_path: Path, *, max_examples: int | None = None
) -> Path:
    rows_out: list[str] = []
    count = 0
    for raw in input_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw)
        nodes = list(row.get("graph", {}).get("nodes", []))
        if len(nodes) < 3:
            continue
        count += 1
        if max_examples is not None and count > max_examples:
            break
        out_row = _build_row(row, idx=count)
        validate_chat_deferred_relevance_payload(out_row)
        rows_out.append(json.dumps(out_row, ensure_ascii=True))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(rows_out), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=".external/argument-mining/processed_data/augmented.jsonl",
    )
    parser.add_argument(
        "--output",
        default="data/arg_mining/chat_deferred_relevance.jsonl",
    )
    parser.add_argument("--max-examples", type=int, default=None)
    args = parser.parse_args()
    path = build_dataset(
        Path(args.input), Path(args.output), max_examples=args.max_examples
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
