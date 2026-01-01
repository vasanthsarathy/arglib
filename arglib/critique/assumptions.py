"""Critique helpers."""

from __future__ import annotations

from arglib.ai.assumptions import generate_edge_assumptions
from arglib.ai.llm import LLMHook
from arglib.core import ArgumentGraph


def suggest_missing_assumptions(
    graph: ArgumentGraph,
    *,
    hook: LLMHook | None = None,
    k: int = 3,
    relation_kinds: tuple[str, ...] = ("support", "attack", "undercut", "rebut"),
) -> list[dict[str, object]]:
    if hook is None:
        return []

    results: list[dict[str, object]] = []
    for rel in graph.relations:
        if rel.kind not in relation_kinds:
            continue
        source = graph.units.get(rel.src)
        target = graph.units.get(rel.dst)
        if not source or not target:
            continue
        assumptions = generate_edge_assumptions(
            source=source.text,
            target=target.text,
            relation=rel.kind,
            k=k,
            hook=hook,
        )
        results.append(
            {
                "src": rel.src,
                "dst": rel.dst,
                "kind": rel.kind,
                "assumptions": [assumption.__dict__ for assumption in assumptions],
            }
        )
    return results
