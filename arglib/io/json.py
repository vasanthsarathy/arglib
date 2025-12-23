"""JSON import/export helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from arglib.core import ArgumentGraph
from arglib.io.schema import validate_graph_payload


def dumps(graph: ArgumentGraph, *, indent: int = 2) -> str:
    return json.dumps(graph.to_dict(), indent=indent, sort_keys=True)


def loads(data: str, *, validate: bool = False) -> ArgumentGraph:
    payload: dict[str, Any] = json.loads(data)
    if validate:
        validate_graph_payload(payload)
    return ArgumentGraph.from_dict(payload)


def save(path: str | Path, graph: ArgumentGraph, *, indent: int = 2) -> None:
    Path(path).write_text(dumps(graph, indent=indent), encoding="utf-8")


def load(path: str | Path, *, validate: bool = False) -> ArgumentGraph:
    payload = Path(path).read_text(encoding="utf-8")
    return loads(payload, validate=validate)
