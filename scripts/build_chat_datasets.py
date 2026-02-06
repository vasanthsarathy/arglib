"""Build starter chat-style datasets from augmented essay graphs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from arglib.io import (
    validate_chat_exploration_payload,
    validate_chat_grounded_payload,
)

_SCENARIOS = ("summarize", "qa", "compare", "debate", "research")


def _normalize_claim_type(value: str) -> str:
    lowered = value.strip().lower()
    if lowered in {"factual", "fact"}:
        return "fact"
    if lowered in {"value", "policy"}:
        return lowered
    return "other"


def _normalize_relation(value: str) -> str:
    lowered = value.strip().lower()
    if lowered in {"support", "supports"}:
        return "support"
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


def _find_span(text: str, needle: str) -> dict[str, int] | None:
    if not needle.strip():
        return None
    idx = text.lower().find(needle.lower().strip())
    if idx < 0:
        return None
    return {"start": idx, "end": idx + len(needle.strip())}


def _sentence_chunks(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _build_grounded_row(
    row: dict[str, Any],
    *,
    idx: int,
) -> dict[str, Any]:
    essay = str(row.get("essay", "")).strip()
    topic = str(row.get("topic", "Unknown topic")).strip() or "Unknown topic"
    issue = str(row.get("issue", "")).strip()
    graph = row.get("graph", {})
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    split = _split_for_index(idx)
    scenario = _SCENARIOS[idx % len(_SCENARIOS)]

    claims = []
    for node in nodes:
        cid = f"c{int(node['id'])}"
        text = str(node.get("text", "")).strip()
        claims.append(
            {
                "id": cid,
                "text": text,
                "type": _normalize_claim_type(str(node.get("type", "other"))),
                "implicit": bool(node.get("implicit", False)),
                "turn_id": "t2",
                "source_span": _find_span(essay, text),
            }
        )

    relations = []
    for edge in edges:
        relations.append(
            {
                "src": f"c{int(edge['from'])}",
                "dst": f"c{int(edge['to'])}",
                "kind": _normalize_relation(str(edge.get("relation", "support"))),
            }
        )

    return {
        "id": f"cg-{idx:05d}",
        "split": split,
        "scenario": scenario,
        "source_documents": [
            {"doc_id": "doc-1", "title": topic, "text": issue},
        ],
        "conversation": [
            {
                "turn_id": "t1",
                "role": "user",
                "text": (
                    "Summarize and evaluate the key arguments from this source. "
                    "Identify major claims and how they support or attack each other."
                ),
            },
            {"turn_id": "t2", "role": "assistant", "text": essay},
        ],
        "labels": {"claims": claims, "relations": relations},
    }


def _build_exploration_row(
    row: dict[str, Any],
    *,
    idx: int,
) -> dict[str, Any]:
    essay = str(row.get("essay", "")).strip()
    topic = str(row.get("topic", "Unknown topic")).strip() or "Unknown topic"
    stance = str(row.get("stance", "")).strip()
    graph = row.get("graph", {})
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    split = _split_for_index(idx)
    scenario = _SCENARIOS[(idx + 2) % len(_SCENARIOS)]

    chunks = _sentence_chunks(essay)
    turn2 = " ".join(chunks[: max(1, len(chunks) // 2)]).strip()
    turn4 = " ".join(chunks[max(1, len(chunks) // 2) :]).strip()
    if not turn4:
        turn4 = turn2

    claims = []
    for node in nodes:
        cid = f"c{int(node['id'])}"
        text = str(node.get("text", "")).strip()
        target_turn = "t2" if _find_span(turn2, text) else "t4"
        span_text = turn2 if target_turn == "t2" else turn4
        claims.append(
            {
                "id": cid,
                "text": text,
                "type": _normalize_claim_type(str(node.get("type", "other"))),
                "implicit": bool(node.get("implicit", False)),
                "turn_id": target_turn,
                "source_span": _find_span(span_text, text),
            }
        )

    relations = []
    for edge in edges:
        relations.append(
            {
                "src": f"c{int(edge['from'])}",
                "dst": f"c{int(edge['to'])}",
                "kind": _normalize_relation(str(edge.get("relation", "support"))),
            }
        )

    return {
        "id": f"ce-{idx:05d}",
        "split": split,
        "scenario": scenario,
        "source_documents": [],
        "conversation": [
            {
                "turn_id": "t1",
                "role": "user",
                "text": (
                    f"I'm exploring arguments about {topic}. "
                    "Give me a first-pass position with supporting claims."
                ),
            },
            {"turn_id": "t2", "role": "assistant", "text": turn2},
            {
                "turn_id": "t3",
                "role": "user",
                "text": "Push back on weak points and refine the argument.",
            },
            {
                "turn_id": "t4",
                "role": "assistant",
                "text": turn4 if turn4 else stance,
            },
        ],
        "labels": {"claims": claims, "relations": relations},
    }


def build_datasets(
    input_path: Path,
    output_dir: Path,
    *,
    max_examples: int | None = None,
) -> tuple[Path, Path]:
    grounded_path = output_dir / "chat_grounded.jsonl"
    exploration_path = output_dir / "chat_exploration.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)

    grounded_rows: list[str] = []
    exploration_rows: list[str] = []

    count = 0
    for raw in input_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw)
        graph = row.get("graph", {})
        if not row.get("essay") or not graph.get("nodes"):
            continue
        count += 1
        if max_examples is not None and count > max_examples:
            break

        grounded = _build_grounded_row(row, idx=count)
        exploration = _build_exploration_row(row, idx=count)
        validate_chat_grounded_payload(grounded)
        validate_chat_exploration_payload(exploration)
        grounded_rows.append(json.dumps(grounded, ensure_ascii=True))
        exploration_rows.append(json.dumps(exploration, ensure_ascii=True))

    grounded_path.write_text("\n".join(grounded_rows), encoding="utf-8")
    exploration_path.write_text("\n".join(exploration_rows), encoding="utf-8")
    return grounded_path, exploration_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=".external/argument-mining/processed_data/augmented.jsonl",
        help="Input augmented JSONL file.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/arg_mining",
        help="Output directory for chat dataset JSONL files.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional cap for quick iteration.",
    )
    args = parser.parse_args()

    grounded, exploration = build_datasets(
        Path(args.input),
        Path(args.output_dir),
        max_examples=args.max_examples,
    )
    print(f"Wrote {grounded}")
    print(f"Wrote {exploration}")


if __name__ == "__main__":
    main()
