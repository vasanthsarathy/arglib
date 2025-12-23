"""Graphviz DOT export helpers."""

from __future__ import annotations

from arglib.core import ArgumentGraph


def to_dot(graph: ArgumentGraph) -> str:
    color_map: dict[str, str] = {
        "support": "green",
        "attack": "red",
        "undercut": "orange",
        "rebut": "blue",
    }

    lines = ["digraph ArgumentGraph {", "  node [shape=box];"]
    for unit_id, unit in graph.units.items():
        label = _escape_label(f"{unit_id}: {unit.text}")
        lines.append(f'  "{unit_id}" [label="{label}"];')

    for relation in graph.relations:
        color = color_map.get(relation.kind, "black")
        label = _escape_label(relation.kind)
        edge = (
            f'  "{relation.src}" -> "{relation.dst}" '
            f'[label="{label}", color="{color}"];'
        )
        lines.append(edge)

    lines.append("}")
    return "\n".join(lines)


def _escape_label(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
