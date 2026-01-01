import json

from arglib.ai import NoOpLLMClient, build_claim_type_hook, classify_claim_type


def test_claim_type_parses_result():
    response = json.dumps(
        {
            "claim_type": "policy",
            "confidence": 0.82,
            "rationale": "It recommends a course of action.",
        }
    )
    hook = build_claim_type_hook(NoOpLLMClient(response=response))

    result = classify_claim_type(
        claim="We should invest in solar energy.",
        hook=hook,
    )

    assert result.claim_type == "policy"
    assert result.confidence == 0.82
    assert result.rationale == "It recommends a course of action."
