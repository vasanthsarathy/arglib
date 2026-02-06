"""Build synthetic grounded-evidence chat dataset for claim-evidence evaluation."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from arglib.io import validate_chat_grounded_evidence_payload

_SCENARIOS = ("summarize", "qa", "compare", "research")


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


def _sentence_chunks(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def _find_span(text: str, needle: str) -> dict[str, int] | None:
    if not needle.strip():
        return None
    idx = text.lower().find(needle.lower().strip())
    if idx < 0:
        return None
    return {"start": idx, "end": idx + len(needle.strip())}


def _negate_sentence(sentence: str) -> str:
    lowered = sentence.lower()
    if " not " in lowered or lowered.startswith("not "):
        return sentence
    if " is " in lowered:
        return sentence.replace(" is ", " is not ", 1)
    if " are " in lowered:
        return sentence.replace(" are ", " are not ", 1)
    if not sentence:
        return sentence
    return f"It is not the case that {sentence[0].lower() + sentence[1:]}"


def _build_row(row: dict[str, Any], *, idx: int) -> dict[str, Any]:
    topic = str(row.get("topic", "Unknown topic")).strip() or "Unknown topic"
    issue = str(row.get("issue", "")).strip()
    essay = str(row.get("essay", "")).strip()
    graph = row.get("graph", {})
    nodes = list(graph.get("nodes", []))
    edges = list(graph.get("edges", []))

    doc_text = issue or essay
    doc_sentences = _sentence_chunks(doc_text)
    if not doc_sentences:
        doc_sentences = _sentence_chunks(essay)
    if not doc_sentences:
        doc_sentences = ["No source sentence available."]

    claims: list[dict[str, Any]] = []
    for node in nodes[:6]:
        claim_text = str(node.get("text", "")).strip()
        if not claim_text:
            continue
        cid = f"c{int(node['id'])}"
        claims.append(
            {
                "id": cid,
                "text": claim_text,
                "type": _normalize_claim_type(str(node.get("type", "other"))),
                "implicit": bool(node.get("implicit", False)),
                "turn_id": "t2",
                "source_span": _find_span(essay, claim_text),
            }
        )
    if not claims:
        claims.append(
            {
                "id": "c1",
                "text": doc_sentences[0],
                "type": "fact",
                "implicit": False,
                "turn_id": "t2",
                "source_span": _find_span(essay, doc_sentences[0]),
            }
        )

    evidence_items: list[dict[str, Any]] = []
    claim_evidence_links: list[dict[str, Any]] = []
    for pos, claim in enumerate(claims, start=1):
        evidence_id = f"e{pos}"
        sentence = doc_sentences[(pos - 1) % len(doc_sentences)]
        span = _find_span(doc_text, sentence)
        evidence_items.append(
            {
                "id": evidence_id,
                "doc_id": "doc-1",
                "text": sentence,
                "doc_span": span,
                "strength": 0.7,
            }
        )
        claim_evidence_links.append(
            {
                "claim_id": claim["id"],
                "evidence_id": evidence_id,
                "stance": "support",
                "confidence": 0.75,
            }
        )

    if claims:
        contradiction_base = claims[0]["text"]
        contradiction_text = _negate_sentence(contradiction_base)
        contradiction_id = f"e{len(evidence_items) + 1}"
        evidence_items.append(
            {
                "id": contradiction_id,
                "doc_id": "doc-1",
                "text": contradiction_text,
                "doc_span": None,
                "strength": 0.55,
            }
        )
        claim_evidence_links.append(
            {
                "claim_id": claims[0]["id"],
                "evidence_id": contradiction_id,
                "stance": "attack",
                "confidence": 0.65,
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

    assistant_text = " ".join(claim["text"] for claim in claims)
    return {
        "id": f"cge-{idx:05d}",
        "split": _split_for_index(idx),
        "scenario": _SCENARIOS[idx % len(_SCENARIOS)],
        "source_documents": [
            {"doc_id": "doc-1", "title": topic, "text": doc_text},
        ],
        "conversation": [
            {
                "turn_id": "t1",
                "role": "user",
                "text": (
                    "Summarize the source and explain supporting and conflicting"
                    " evidence."
                ),
            },
            {"turn_id": "t2", "role": "assistant", "text": assistant_text},
        ],
        "labels": {
            "claims": claims,
            "relations": relations,
            "evidence_items": evidence_items,
            "claim_evidence_links": claim_evidence_links,
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
        graph = row.get("graph", {})
        if not graph.get("nodes"):
            continue
        count += 1
        if max_examples is not None and count > max_examples:
            break
        out_row = _build_row(row, idx=count)
        validate_chat_grounded_evidence_payload(out_row)
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
        default="data/arg_mining/chat_grounded_evidence.jsonl",
    )
    parser.add_argument("--max-examples", type=int, default=None)
    args = parser.parse_args()
    path = build_dataset(
        Path(args.input), Path(args.output), max_examples=args.max_examples
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
