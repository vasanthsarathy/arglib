import json

from arglib.io import (
    validate_chat_exploration_dict,
    validate_chat_exploration_payload,
    validate_chat_grounded_dict,
    validate_chat_grounded_evidence_dict,
    validate_chat_grounded_evidence_payload,
    validate_chat_grounded_payload,
)


def _grounded_row() -> dict:
    return {
        "id": "cg-00001",
        "split": "train",
        "scenario": "summarize",
        "source_documents": [
            {"doc_id": "doc-1", "title": "Topic", "text": "Source text."}
        ],
        "conversation": [
            {"turn_id": "t1", "role": "user", "text": "Summarize this."},
            {"turn_id": "t2", "role": "assistant", "text": "Claim A. Claim B."},
        ],
        "labels": {
            "claims": [
                {
                    "id": "c1",
                    "text": "Claim A",
                    "type": "fact",
                    "turn_id": "t2",
                    "source_span": {"start": 0, "end": 7},
                },
                {
                    "id": "c2",
                    "text": "Claim B",
                    "type": "policy",
                    "turn_id": "t2",
                    "source_span": {"start": 9, "end": 16},
                },
            ],
            "relations": [{"src": "c1", "dst": "c2", "kind": "support"}],
        },
    }


def _exploration_row() -> dict:
    return {
        "id": "ce-00001",
        "split": "dev",
        "scenario": "debate",
        "source_documents": [],
        "conversation": [
            {"turn_id": "t1", "role": "user", "text": "Take a stance."},
            {"turn_id": "t2", "role": "assistant", "text": "First argument."},
            {"turn_id": "t3", "role": "user", "text": "Challenge it."},
            {"turn_id": "t4", "role": "assistant", "text": "Refined argument."},
        ],
        "labels": {
            "claims": [
                {
                    "id": "c1",
                    "text": "First argument",
                    "type": "value",
                    "turn_id": "t2",
                    "source_span": {"start": 0, "end": 14},
                },
                {
                    "id": "c2",
                    "text": "Refined argument",
                    "type": "policy",
                    "turn_id": "t4",
                    "source_span": {"start": 0, "end": 16},
                },
            ],
            "relations": [{"src": "c2", "dst": "c1", "kind": "support"}],
        },
    }


def _grounded_evidence_row() -> dict:
    row = _grounded_row()
    row["id"] = "cge-00001"
    row["labels"]["evidence_items"] = [
        {
            "id": "e1",
            "doc_id": "doc-1",
            "text": "Source text.",
            "doc_span": {"start": 0, "end": 12},
            "strength": 0.8,
        }
    ]
    row["labels"]["claim_evidence_links"] = [
        {
            "claim_id": "c1",
            "evidence_id": "e1",
            "stance": "support",
            "confidence": 0.9,
        }
    ]
    return row


def test_chat_grounded_validation_ok():
    row = _grounded_row()
    assert validate_chat_grounded_dict(row) == []
    validate_chat_grounded_payload(row)


def test_chat_exploration_validation_ok():
    row = _exploration_row()
    assert validate_chat_exploration_dict(row) == []
    validate_chat_exploration_payload(row)


def test_chat_grounded_evidence_validation_ok():
    row = _grounded_evidence_row()
    assert validate_chat_grounded_evidence_dict(row) == []
    validate_chat_grounded_evidence_payload(row)


def test_chat_dataset_validation_rejects_bad_relation():
    row = _grounded_row()
    row["labels"]["relations"] = [{"src": "c1", "dst": "c9", "kind": "support"}]
    errors = validate_chat_grounded_dict(row)
    payload = json.dumps(errors)
    assert "unknown claim" in payload


def test_chat_grounded_evidence_rejects_bad_link():
    row = _grounded_evidence_row()
    row["labels"]["claim_evidence_links"] = [
        {"claim_id": "c9", "evidence_id": "e1", "stance": "support"}
    ]
    errors = validate_chat_grounded_evidence_dict(row)
    assert any("must reference known claim" in error for error in errors)
