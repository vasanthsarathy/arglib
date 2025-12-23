import json
import pytest

from arglib.core import ArgumentGraph
from arglib.io import loads, validate_graph_dict, validate_graph_payload


def test_validate_graph_payload_ok():
    graph = ArgumentGraph.new()
    a = graph.add_claim("A")
    b = graph.add_claim("B")
    graph.add_attack(a, b)

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
