import json

from scripts.eval_deferred_relevance import evaluate


def test_eval_deferred_relevance_runs(tmp_path):
    row = {
        "id": "cd-1",
        "split": "train",
        "scenario": "research",
        "source_documents": [],
        "conversation": [
            {"turn_id": "t1", "role": "user", "text": "Topic."},
            {
                "turn_id": "t2",
                "role": "assistant",
                "text": "Main point. Side note: X.",
            },
            {"turn_id": "t3", "role": "user", "text": "Why is that relevant?"},
            {
                "turn_id": "t4",
                "role": "assistant",
                "text": "To clarify the earlier side note, X supports main point.",
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
                    }
                ],
            },
        },
    }
    dataset = tmp_path / "deferred.jsonl"
    dataset.write_text(json.dumps(row), encoding="utf-8")
    report = evaluate(dataset)
    assert report["rows"] == 1
    assert "pending_detection" in report
