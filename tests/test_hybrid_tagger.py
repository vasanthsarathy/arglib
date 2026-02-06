import json

from arglib.ai import HybridClaimRelationTagger


def test_hybrid_tagger_falls_back_when_model_missing():
    tagger = HybridClaimRelationTagger(
        model_path="models/does_not_exist.json",
        neural_model_dir="models/does_not_exist_neural",
    )
    result = tagger.tag("The model is useful. Therefore we should use it.")
    assert result.claims
    assert result.graph.metadata.get("tagger_mode") == "fallback"


def test_hybrid_tagger_uses_small_model(tmp_path):
    model_path = tmp_path / "small.json"
    model_payload = {
        "claim": {
            "labels": ["fact", "value", "policy", "other"],
            "alpha": 1.0,
            "priors": {"fact": 0.0, "value": -2.0, "policy": -1.0, "other": -3.0},
            "token_log_probs": {
                "fact": {"data": 0.0},
                "value": {"data": -1.0},
                "policy": {"data": -1.0, "should": 0.0},
                "other": {"data": -2.0},
            },
            "unk_log_prob": {
                "fact": -2.0,
                "value": -2.0,
                "policy": -2.0,
                "other": -2.0,
            },
        },
        "relation": {
            "labels": ["support", "attack"],
            "alpha": 1.0,
            "priors": {"support": 0.0, "attack": -1.0},
            "token_log_probs": {
                "support": {"jaccard_bin:2": 0.0},
                "attack": {"neg_flip:1": 0.0},
            },
            "unk_log_prob": {"support": -1.0, "attack": -1.0},
        },
        "evidence_stance": {
            "labels": ["support", "attack", "insufficient"],
            "alpha": 1.0,
            "priors": {"support": 0.0, "attack": -1.0, "insufficient": -1.0},
            "token_log_probs": {
                "support": {"jaccard_bin:2": 0.0},
                "attack": {"neg_flip:1": 0.0},
                "insufficient": {"jaccard_bin:0": 0.0},
            },
            "unk_log_prob": {
                "support": -1.0,
                "attack": -1.0,
                "insufficient": -1.0,
            },
        },
        "evidence_retrieval": {
            "labels": ["yes", "no"],
            "alpha": 1.0,
            "priors": {"yes": 0.0, "no": -1.0},
            "token_log_probs": {
                "yes": {"jaccard_bin:2": 0.0},
                "no": {"jaccard_bin:0": 0.0},
            },
            "unk_log_prob": {"yes": -1.0, "no": -1.0},
        },
    }
    model_path.write_text(json.dumps(model_payload), encoding="utf-8")
    tagger = HybridClaimRelationTagger(model_path=str(model_path))
    result = tagger.tag("Data is reliable; we should use data.")
    assert result.claims
    assert result.graph.metadata.get("tagger_mode") == "hybrid"

    links = tagger.predict_evidence_links(
        claims=result.claims,
        evidence_items=[{"id": "e1", "text": "Data is reliable"}],
    )
    assert links
