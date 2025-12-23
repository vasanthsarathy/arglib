import json

import pytest

from arglib.core import ArgumentGraph, EvidenceCard, SupportingDocument
from arglib.io import loads, validate_graph_dict, validate_graph_payload


def test_validate_graph_payload_ok():
    graph = ArgumentGraph.new()
    a = graph.add_claim("A")
    b = graph.add_claim("B")
    graph.add_attack(a, b)
    graph.add_supporting_document(
        SupportingDocument(id="doc1", name="Doc", type="pdf", url="file://doc1")
    )
    graph.add_evidence_card(
        EvidenceCard(
            id="e1",
            title="Evidence",
            supporting_doc_id="doc1",
            excerpt="...",
            confidence=0.9,
        )
    )
    graph.attach_evidence_card(a, "e1")

    payload = graph.to_dict()
    errors = validate_graph_dict(payload)
    assert errors == []

    validate_graph_payload(payload)


def test_validate_graph_payload_unknown_relation():
    graph = ArgumentGraph.new()
    graph.add_claim("A")
    payload = graph.to_dict()
    payload["relations"].append({"src": "c99", "dst": "c1", "kind": "attack"})

    with pytest.raises(ValueError) as excinfo:
        validate_graph_payload(payload)

    assert "not a known unit id" in str(excinfo.value)


def test_loads_strict_validation():
    graph = ArgumentGraph.new()
    payload = graph.to_dict()
    payload["relations"].append({"src": "c9", "dst": "c1", "kind": "attack"})

    with pytest.raises(ValueError):
        loads(json.dumps(payload), validate=True)
