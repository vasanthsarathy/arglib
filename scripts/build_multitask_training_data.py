"""Build task-specific JSONL shards for multitask training."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

TASKS = (
    "span_tagging",
    "claim_type",
    "relation_class",
    "coref_link",
    "deferred_pending",
    "deferred_resolution",
    "discourse_function",
    "contradiction_nli",
    "claim_evidence_stance",
    "evidence_retrieval",
)


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _find_span(text: str, phrase: str) -> dict[str, int] | None:
    idx = text.lower().find(phrase.lower())
    if idx < 0:
        return None
    return {"start": idx, "end": idx + len(phrase)}


def _bio_tags(text: str, spans: list[dict[str, int]]) -> list[dict[str, Any]]:
    # Character-level tags keep this framework/model agnostic.
    tags = ["O"] * len(text)
    for span in spans:
        start = max(0, int(span["start"]))
        end = min(len(text), int(span["end"]))
        if end <= start:
            continue
        tags[start] = "B-CLAIM"
        for idx in range(start + 1, end):
            tags[idx] = "I-CLAIM"
    return [{"char": text[idx], "tag": tags[idx]} for idx in range(len(text))]


def _default_discourse(role: str, text: str) -> str:
    if role == "user":
        return "question"
    lowered = text.lower()
    if "to clarify" in lowered or "because" in lowered:
        return "explanation"
    if "side note" in lowered:
        return "context"
    return "claim"


def build(input_paths: list[Path], output_dir: Path) -> dict[str, int]:
    shards: dict[str, list[dict[str, Any]]] = {task: [] for task in TASKS}

    for path in input_paths:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for row in rows:
            split = str(row.get("split", "train"))
            conversation = [
                turn for turn in row.get("conversation", []) if isinstance(turn, dict)
            ]
            labels = row.get("labels", {})
            claims = [
                claim
                for claim in labels.get("claims", [])
                if isinstance(claim, dict)
            ]
            relations = [
                rel for rel in labels.get("relations", []) if isinstance(rel, dict)
            ]
            evidence_items = [
                item
                for item in labels.get("evidence_items", [])
                if isinstance(item, dict)
            ]
            claim_evidence_links = [
                item
                for item in labels.get("claim_evidence_links", [])
                if isinstance(item, dict)
            ]
            claims_by_turn: dict[str, list[dict[str, Any]]] = defaultdict(list)
            claim_by_id: dict[str, dict[str, Any]] = {}
            for claim in claims:
                cid = str(claim.get("id", ""))
                claim_by_id[cid] = claim
                claims_by_turn[str(claim.get("turn_id", ""))].append(claim)

            for turn in conversation:
                turn_id = str(turn.get("turn_id", ""))
                role = str(turn.get("role", "assistant"))
                text = str(turn.get("text", ""))
                turn_claims = claims_by_turn.get(turn_id, [])
                spans = []
                for claim in turn_claims:
                    span = claim.get("source_span")
                    if isinstance(span, dict):
                        spans.append(span)
                        continue
                    found = _find_span(text, str(claim.get("text", "")))
                    if found is not None:
                        spans.append(found)
                shards["span_tagging"].append(
                    {
                        "split": split,
                        "row_id": row.get("id"),
                        "turn_id": turn_id,
                        "text": text,
                        "tags": _bio_tags(text, spans),
                    }
                )
                for claim in turn_claims:
                    shards["claim_type"].append(
                        {
                            "split": split,
                            "row_id": row.get("id"),
                            "turn_id": turn_id,
                            "claim_id": claim.get("id"),
                            "text": claim.get("text"),
                            "label": claim.get("type", "other"),
                        }
                    )
                    deferred = labels.get("deferred_relevance", {})
                    pending_items = deferred.get("pending_items", [])
                    pending_ids = {
                        str(item.get("claim_id"))
                        for item in pending_items
                        if isinstance(item, dict)
                    }
                    shards["deferred_pending"].append(
                        {
                            "split": split,
                            "row_id": row.get("id"),
                            "claim_id": claim.get("id"),
                            "text": claim.get("text"),
                            "label": "pending"
                            if str(claim.get("id")) in pending_ids
                            else "not_pending",
                        }
                    )
                discourse = turn.get("discourse_function")
                if not discourse:
                    discourse = _default_discourse(role, text)
                shards["discourse_function"].append(
                    {
                        "split": split,
                        "row_id": row.get("id"),
                        "turn_id": turn_id,
                        "text": text,
                        "label": discourse,
                    }
                )

            rel_set = {
                (str(rel.get("src")), str(rel.get("dst"))): str(
                    rel.get("kind", "support")
                )
                for rel in relations
            }
            claim_ids = [str(claim.get("id")) for claim in claims if claim.get("id")]
            for src, dst in combinations(claim_ids, 2):
                pairs = [(src, dst), (dst, src)]
                for left, right in pairs:
                    label = rel_set.get((left, right), "none")
                    shards["relation_class"].append(
                        {
                            "split": split,
                            "row_id": row.get("id"),
                            "src_id": left,
                            "dst_id": right,
                            "src_text": claim_by_id[left].get("text", ""),
                            "dst_text": claim_by_id[right].get("text", ""),
                            "label": label,
                        }
                    )
                    shards["contradiction_nli"].append(
                        {
                            "split": split,
                            "row_id": row.get("id"),
                            "premise": claim_by_id[left].get("text", ""),
                            "hypothesis": claim_by_id[right].get("text", ""),
                            "label": (
                                "contradicts"
                                if label == "attack"
                                else "entails"
                                if label == "support"
                                else "neutral"
                            ),
                        }
                    )

            evidence_by_id = {
                str(item.get("id")): item for item in evidence_items if item.get("id")
            }
            positive_links: set[tuple[str, str]] = set()
            for link in claim_evidence_links:
                claim_id = str(link.get("claim_id", ""))
                evidence_id = str(link.get("evidence_id", ""))
                stance = str(link.get("stance", "insufficient"))
                if claim_id not in claim_by_id or evidence_id not in evidence_by_id:
                    continue
                positive_links.add((claim_id, evidence_id))
                shards["claim_evidence_stance"].append(
                    {
                        "split": split,
                        "row_id": row.get("id"),
                        "claim_id": claim_id,
                        "evidence_id": evidence_id,
                        "claim_text": claim_by_id[claim_id].get("text", ""),
                        "evidence_text": evidence_by_id[evidence_id].get("text", ""),
                        "label": stance,
                    }
                )
            for claim_id in claim_ids:
                for evidence_id, item in evidence_by_id.items():
                    label = "yes" if (claim_id, evidence_id) in positive_links else "no"
                    shards["evidence_retrieval"].append(
                        {
                            "split": split,
                            "row_id": row.get("id"),
                            "claim_id": claim_id,
                            "evidence_id": evidence_id,
                            "claim_text": claim_by_id[claim_id].get("text", ""),
                            "evidence_text": item.get("text", ""),
                            "label": label,
                        }
                    )

            deferred = labels.get("deferred_relevance", {})
            pending_items = [
                item
                for item in deferred.get("pending_items", [])
                if isinstance(item, dict)
            ]
            resolution_links = [
                item
                for item in deferred.get("resolution_links", [])
                if isinstance(item, dict)
            ]
            for link in resolution_links:
                pending_id = str(link.get("pending_claim_id", ""))
                target_id = str(link.get("target_claim_id", ""))
                if pending_id not in claim_by_id or target_id not in claim_by_id:
                    continue
                shards["deferred_resolution"].append(
                    {
                        "split": split,
                        "row_id": row.get("id"),
                        "pending_claim_id": pending_id,
                        "target_claim_id": target_id,
                        "resolver_turn": link.get("resolver_turn"),
                        "pending_text": claim_by_id[pending_id].get("text", ""),
                        "target_text": claim_by_id[target_id].get("text", ""),
                        "label": "resolve",
                    }
                )
                shards["coref_link"].append(
                    {
                        "split": split,
                        "row_id": row.get("id"),
                        "source_claim_id": pending_id,
                        "target_claim_id": target_id,
                        "source_text": claim_by_id[pending_id].get("text", ""),
                        "target_text": claim_by_id[target_id].get("text", ""),
                        "label": "yes",
                    }
                )
            pending_ids = [str(item.get("claim_id", "")) for item in pending_items]
            for pending_id in pending_ids:
                if pending_id not in claim_by_id:
                    continue
                for candidate_id in claim_ids:
                    if candidate_id == pending_id:
                        continue
                    if any(
                        str(link.get("pending_claim_id")) == pending_id
                        and str(link.get("target_claim_id")) == candidate_id
                        for link in resolution_links
                    ):
                        continue
                    shards["coref_link"].append(
                        {
                            "split": split,
                            "row_id": row.get("id"),
                            "source_claim_id": pending_id,
                            "target_claim_id": candidate_id,
                            "source_text": claim_by_id[pending_id].get("text", ""),
                            "target_text": claim_by_id[candidate_id].get("text", ""),
                            "label": "no",
                        }
                    )
                    shards["deferred_resolution"].append(
                        {
                            "split": split,
                            "row_id": row.get("id"),
                            "pending_claim_id": pending_id,
                            "target_claim_id": candidate_id,
                            "resolver_turn": None,
                            "pending_text": claim_by_id[pending_id].get("text", ""),
                            "target_text": claim_by_id[candidate_id].get("text", ""),
                            "label": "none",
                        }
                    )

    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for task, rows in shards.items():
        path = output_dir / f"{task}.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        counts[task] = len(rows)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "data/arg_mining/chat_grounded.jsonl",
            "data/arg_mining/chat_exploration.jsonl",
            "data/arg_mining/chat_deferred_relevance.jsonl",
            "data/arg_mining/chat_grounded_evidence.jsonl",
        ],
    )
    parser.add_argument("--output-dir", default="data/training/multitask")
    args = parser.parse_args()

    counts = build([Path(path) for path in args.inputs], Path(args.output_dir))
    print(json.dumps({"output_dir": args.output_dir, "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
