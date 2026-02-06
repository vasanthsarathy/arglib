"""Evaluate tagger robustness under synthetic text perturbations."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable
from pathlib import Path

from arglib.ai import ClaimRelationTagger
from arglib.ai.mining import token_jaccard_similarity

Perturbation = Callable[[str], str]


def _assistant_text(row: dict) -> str:
    turns = row.get("conversation", [])
    chunks = [
        str(turn.get("text", "")).strip()
        for turn in turns
        if isinstance(turn, dict) and str(turn.get("role")) == "assistant"
    ]
    return " ".join(chunk for chunk in chunks if chunk)


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def perturb_paraphrase(text: str) -> str:
    replacements = {
        "Therefore,": "So,",
        "therefore,": "so,",
        "However,": "But,",
        "however,": "but,",
        "This is because": "This happens because",
        "This is crucial": "This matters",
    }
    updated = text
    for src, dst in replacements.items():
        updated = updated.replace(src, dst)
    return updated


def perturb_reorder(text: str) -> str:
    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        return text
    return " ".join(reversed(sentences))


def perturb_negation(text: str) -> str:
    sentences = _split_sentences(text)
    flipped: list[str] = []
    pattern = re.compile(r"\b(is|are|was|were|has|have|can|will|should|must)\b", re.I)
    for sentence in sentences:
        if re.search(r"\bnot\b", sentence, re.I):
            sentence = re.sub(r"\bnot\s+", "", sentence, flags=re.I)
            flipped.append(sentence)
            continue
        match = pattern.search(sentence)
        if not match:
            flipped.append(f"Not {sentence[0].lower()}{sentence[1:]}")
            continue
        idx = match.end()
        updated = sentence[:idx] + " not" + sentence[idx:]
        flipped.append(updated)
    return " ".join(flipped)


def _align(base_claims: list[str], other_claims: list[str]) -> list[tuple[int, int]]:
    candidates: list[tuple[float, int, int]] = []
    for bi, base in enumerate(base_claims):
        for oi, other in enumerate(other_claims):
            score = token_jaccard_similarity(base, other)
            if score >= 0.45:
                candidates.append((score, bi, oi))
    candidates.sort(reverse=True)
    used_b: set[int] = set()
    used_o: set[int] = set()
    aligned: list[tuple[int, int]] = []
    for _score, bi, oi in candidates:
        if bi in used_b or oi in used_o:
            continue
        aligned.append((bi, oi))
        used_b.add(bi)
        used_o.add(oi)
    return aligned


def _f1(tp: int, pred: int, gold: int) -> float:
    precision = tp / pred if pred else 0.0
    recall = tp / gold if gold else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate_robustness(path: Path, *, max_rows: int | None = None) -> dict:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if max_rows is not None:
        rows = rows[:max_rows]

    tagger = ClaimRelationTagger()
    perturbations: dict[str, Perturbation] = {
        "paraphrase": perturb_paraphrase,
        "reorder": perturb_reorder,
        "negation": perturb_negation,
    }

    summary: dict[str, dict[str, float]] = {
        name: {
            "claim_f1_sum": 0.0,
            "type_agreement_sum": 0.0,
            "relation_f1_sum": 0.0,
            "attack_rate_delta_sum": 0.0,
            "rows": 0.0,
        }
        for name in perturbations
    }

    for row in rows:
        base_text = _assistant_text(row)
        if not base_text:
            continue
        base = tagger.tag(base_text, doc_id="base")
        base_claim_texts = [c.text for c in base.claims]
        base_claim_types = [c.claim_type for c in base.claims]
        base_rel_set = {(r.src, r.dst, r.kind) for r in base.relations}
        base_attack_rate = (
            sum(1 for rel in base.relations if rel.kind == "attack")
            / len(base.relations)
            if base.relations
            else 0.0
        )

        for name, perturb in perturbations.items():
            perturbed_text = perturb(base_text)
            pred = tagger.tag(perturbed_text, doc_id=f"perturbed-{name}")
            pred_claim_texts = [c.text for c in pred.claims]
            pred_claim_types = [c.claim_type for c in pred.claims]
            aligned = _align(base_claim_texts, pred_claim_texts)

            claim_f1 = _f1(len(aligned), len(pred_claim_texts), len(base_claim_texts))

            if aligned:
                type_matches = sum(
                    1
                    for bi, oi in aligned
                    if base_claim_types[bi] == pred_claim_types[oi]
                )
                type_agreement = type_matches / len(aligned)
            else:
                type_agreement = 0.0

            aligned_base_ids = {f"c{bi + 1}": f"c{oi + 1}" for bi, oi in aligned}
            pred_rel_mapped = set()
            for src, dst, kind in {(r.src, r.dst, r.kind) for r in pred.relations}:
                inv_src = None
                inv_dst = None
                for b_id, p_id in aligned_base_ids.items():
                    if p_id == src:
                        inv_src = b_id
                    if p_id == dst:
                        inv_dst = b_id
                if inv_src is None or inv_dst is None:
                    continue
                pred_rel_mapped.add((inv_src, inv_dst, kind))
            relation_f1 = _f1(
                len(base_rel_set & pred_rel_mapped),
                len(pred_rel_mapped),
                len(base_rel_set),
            )
            pred_attack_rate = (
                sum(1 for rel in pred.relations if rel.kind == "attack")
                / len(pred.relations)
                if pred.relations
                else 0.0
            )

            stats = summary[name]
            stats["claim_f1_sum"] += claim_f1
            stats["type_agreement_sum"] += type_agreement
            stats["relation_f1_sum"] += relation_f1
            stats["attack_rate_delta_sum"] += abs(pred_attack_rate - base_attack_rate)
            stats["rows"] += 1.0

    out: dict[str, dict[str, float]] = {}
    for name, stats in summary.items():
        rows_used = stats["rows"] if stats["rows"] else 1.0
        out[name] = {
            "rows": stats["rows"],
            "claim_retention_f1": stats["claim_f1_sum"] / rows_used,
            "type_agreement": stats["type_agreement_sum"] / rows_used,
            "relation_overlap_f1": stats["relation_f1_sum"] / rows_used,
            "avg_attack_rate_delta": stats["attack_rate_delta_sum"] / rows_used,
        }

    return {"dataset": str(path), "rows_evaluated": len(rows), "perturbations": out}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Path to dataset JSONL.")
    parser.add_argument("--output", required=True, help="Path to report JSON output.")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap for quick iteration.",
    )
    args = parser.parse_args()

    report = evaluate_robustness(Path(args.dataset), max_rows=args.max_rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved report: {output}")


if __name__ == "__main__":
    main()
