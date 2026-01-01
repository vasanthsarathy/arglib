import json

from arglib.ai import NoOpLLMClient, build_edge_validation_hook, validate_edge_with_llm


def test_edge_validation_parses_result():
    response = json.dumps(
        {
            "evaluation": "support",
            "score": 0.7,
            "rationale": "Direct entailment.",
        }
    )
    hook = build_edge_validation_hook(NoOpLLMClient(response=response))

    result = validate_edge_with_llm(
        source="All mammals are warm-blooded.",
        target="Dogs are warm-blooded.",
        hook=hook,
    )

    assert result.evaluation == "support"
    assert result.score == 0.7
    assert result.rationale == "Direct entailment."
