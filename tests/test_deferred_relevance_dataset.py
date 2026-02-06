from arglib.io import (
    validate_chat_deferred_relevance_dict,
    validate_chat_deferred_relevance_payload,
)


def test_deferred_relevance_validation_ok():
    row = {
        "id": "cd-00001",
        "split": "train",
        "scenario": "research",
        "source_documents": [],
        "conversation": [
            {
                "turn_id": "t1",
                "role": "user",
                "text": "Discuss this.",
                "discourse_function": "question",
            },
            {
                "turn_id": "t2",
                "role": "assistant",
                "text": "Main point. Side note: X.",
                "discourse_function": "claim",
            },
            {
                "turn_id": "t4",
                "role": "assistant",
                "text": "To clarify the earlier side note, X supports main point.",
                "discourse_function": "explanation",
            },
        ],
        "labels": {
            "claims": [
                {"id": "c1", "text": "X", "type": "fact", "turn_id": "t2"},
                {"id": "c2", "text": "Main point", "type": "policy", "turn_id": "t2"},
            ],
            "relations": [{"src": "c1", "dst": "c2", "kind": "support"}],
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
                    }
                ],
            },
        },
    }
    assert validate_chat_deferred_relevance_dict(row) == []
    validate_chat_deferred_relevance_payload(row)
