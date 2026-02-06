import json
from pathlib import Path

from arglib.io import validate_chat_grounded_evidence_payload
from scripts.build_grounded_evidence_dataset import build_dataset
from scripts.build_multitask_training_data import build


def test_build_grounded_evidence_dataset(tmp_path: Path):
    input_path = tmp_path / "augmented.jsonl"
    input_row = {
        "topic": "Urban policy",
        "issue": "Parks reduce heat. Parks improve health.",
        "essay": "Parks reduce heat. Therefore cities should fund parks.",
        "graph": {
            "nodes": [
                {"id": 1, "text": "Parks reduce heat", "type": "factual"},
                {"id": 2, "text": "Cities should fund parks", "type": "policy"},
            ],
            "edges": [{"from": 1, "to": 2, "relation": "support"}],
        },
    }
    input_path.write_text(json.dumps(input_row), encoding="utf-8")
    output_path = tmp_path / "chat_grounded_evidence.jsonl"
    build_dataset(input_path, output_path)

    rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows
    validate_chat_grounded_evidence_payload(rows[0])
    assert rows[0]["labels"]["evidence_items"]
    assert rows[0]["labels"]["claim_evidence_links"]


def test_multitask_builder_emits_evidence_tasks(tmp_path: Path):
    dataset = tmp_path / "chat_grounded_evidence.jsonl"
    row = {
        "id": "cge-1",
        "split": "train",
        "scenario": "summarize",
        "source_documents": [
            {"doc_id": "doc-1", "title": "T", "text": "Evidence text."}
        ],
        "conversation": [
            {"turn_id": "t1", "role": "user", "text": "Summarize."},
            {"turn_id": "t2", "role": "assistant", "text": "Claim text."},
        ],
        "labels": {
            "claims": [
                {"id": "c1", "text": "Claim text", "type": "fact", "turn_id": "t2"}
            ],
            "relations": [],
            "evidence_items": [
                {"id": "e1", "doc_id": "doc-1", "text": "Evidence text"}
            ],
            "claim_evidence_links": [
                {"claim_id": "c1", "evidence_id": "e1", "stance": "support"}
            ],
        },
    }
    dataset.write_text(json.dumps(row), encoding="utf-8")
    counts = build([dataset], tmp_path / "multitask")
    assert counts["claim_evidence_stance"] >= 1
    assert counts["evidence_retrieval"] >= 1
