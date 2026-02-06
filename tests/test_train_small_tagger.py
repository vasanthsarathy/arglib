import json

from arglib.ai.small_model import (
    CLAIM_LABELS,
    RELATION_LABELS,
    NaiveBayesTextClassifier,
    relation_features,
)
from scripts.train_small_tagger import train_and_evaluate


def test_naive_bayes_text_learns_simple_boundary():
    model = NaiveBayesTextClassifier(labels=CLAIM_LABELS)
    train = [
        (["should", "must", "policy"], "policy"),
        (["good", "bad", "ethical"], "value"),
        (["is", "are", "data"], "fact"),
        (["misc"], "other"),
    ]
    model.fit(train)
    pred, _ = model.predict(["should", "policy"])
    assert pred == "policy"


def test_relation_features_include_negation_signal():
    features = relation_features("The model is reliable", "The model is not reliable")
    assert "neg_flip:1" in features
    rel_model = NaiveBayesTextClassifier(labels=RELATION_LABELS)
    rel_model.fit(
        [
            (relation_features("A is true", "A is not true"), "attack"),
            (relation_features("A is true", "A is true"), "support"),
        ]
    )
    pred, _ = rel_model.predict(
        relation_features("X is useful", "X is not useful")
    )
    assert pred in {"attack", "support"}


def test_train_and_evaluate_includes_evidence_heads(tmp_path):
    dataset = tmp_path / "synthetic.jsonl"
    rows = [
        {
            "split": "train",
            "labels": {
                "claims": [
                    {"id": "c1", "text": "A is true", "type": "fact"},
                    {"id": "c2", "text": "A is useful", "type": "value"},
                ],
                "relations": [{"src": "c1", "dst": "c2", "kind": "support"}],
                "evidence_items": [{"id": "e1", "text": "A is true in docs"}],
                "claim_evidence_links": [
                    {"claim_id": "c1", "evidence_id": "e1", "stance": "support"}
                ],
            },
        },
        {
            "split": "dev",
            "labels": {
                "claims": [{"id": "c3", "text": "A is not true", "type": "fact"}],
                "relations": [],
                "evidence_items": [{"id": "e2", "text": "A is not true"}],
                "claim_evidence_links": [
                    {"claim_id": "c3", "evidence_id": "e2", "stance": "support"}
                ],
            },
        },
        {
            "split": "test",
            "labels": {
                "claims": [
                    {
                        "id": "c4",
                        "text": "Policy should change",
                        "type": "policy",
                    }
                ],
                "relations": [],
                "evidence_items": [{"id": "e3", "text": "Policy should change"}],
                "claim_evidence_links": [
                    {"claim_id": "c4", "evidence_id": "e3", "stance": "support"}
                ],
            },
        },
    ]
    dataset.write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )
    report = train_and_evaluate([dataset])
    assert report["train_sizes"]["evidence_stance"] >= 1
    assert report["train_sizes"]["evidence_retrieval"] >= 1
    assert "evidence_stance" in report["splits"]["dev"]
    assert "evidence_retrieval" in report["splits"]["test"]
