from arglib.io import (
    validate_chat_exploration_payload,
    validate_chat_grounded_evidence_payload,
    validate_chat_grounded_payload,
)


def _row() -> dict:
    return {
        "id": "cg-1",
        "split": "train",
        "scenario": "summarize",
        "source_documents": [{"doc_id": "d1", "title": "t", "text": "x"}],
        "conversation": [
            {"turn_id": "t1", "role": "user", "text": "Summarize."},
            {"turn_id": "t2", "role": "assistant", "text": "A. Therefore B."},
        ],
        "labels": {
            "claims": [
                {"id": "c1", "text": "A", "type": "fact", "turn_id": "t2"},
                {"id": "c2", "text": "B", "type": "policy", "turn_id": "t2"},
            ],
            "relations": [{"src": "c1", "dst": "c2", "kind": "support"}],
        },
    }


def test_payload_validators_accept_minimal_rows():
    grounded = _row()
    exploration = {**_row(), "id": "ce-1", "source_documents": []}
    grounded_evidence = {
        **_row(),
        "id": "cge-1",
        "labels": {
            **_row()["labels"],
            "evidence_items": [{"id": "e1", "doc_id": "d1", "text": "x"}],
            "claim_evidence_links": [
                {"claim_id": "c1", "evidence_id": "e1", "stance": "support"}
            ],
        },
    }
    validate_chat_grounded_payload(grounded)
    validate_chat_exploration_payload(exploration)
    validate_chat_grounded_evidence_payload(grounded_evidence)
