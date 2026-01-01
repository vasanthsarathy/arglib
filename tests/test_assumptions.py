import json

from arglib.ai import NoOpLLMClient, build_assumption_hook, generate_edge_assumptions


def test_generate_edge_assumptions_parses_items():
    response = json.dumps(
        {
            "assumptions": [
                {
                    "assumption": "All birds are animals.",
                    "rationale": "Needed to link species to category.",
                    "importance": 0.8,
                },
                {
                    "assumption": "Penguins are birds.",
                    "rationale": "Establishes membership.",
                    "importance": 0.7,
                },
            ]
        }
    )
    hook = build_assumption_hook(NoOpLLMClient(response=response))

    assumptions = generate_edge_assumptions(
        source="Penguins are animals.",
        target="Penguins are birds.",
        relation="support",
        k=2,
        hook=hook,
    )

    assert len(assumptions) == 2
    assert assumptions[0].assumption == "All birds are animals."
    assert assumptions[0].importance == 0.8
    assert assumptions[1].assumption == "Penguins are birds."


def test_generate_edge_assumptions_handles_list_response():
    response = json.dumps(
        [
            {
                "assumption": "Tax policy affects disposable income.",
                "rationale": "Links policy to household finances.",
                "importance": 0.5,
            }
        ]
    )
    hook = build_assumption_hook(NoOpLLMClient(response=response))

    assumptions = generate_edge_assumptions(
        source="Tax cuts increase consumption.",
        target="Disposable income rises after tax cuts.",
        relation="support",
        k=3,
        hook=hook,
    )

    assert len(assumptions) == 1
    assert assumptions[0].assumption == "Tax policy affects disposable income."
