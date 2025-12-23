"""Unified reasoning interface."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from arglib.core import ArgumentGraph


class Reasoner:
    def __init__(self, graph: ArgumentGraph) -> None:
        self.graph = graph

    def run(self, tasks: Iterable[str], explain: bool = False) -> dict[str, Any]:
        af = self.graph.to_dung()
        results: dict[str, Any] = {}

        for task in tasks:
            task_key = task.lower()
            if task_key == "grounded_extension":
                results[task] = _normalize_extension(af.grounded_extension())
            elif task_key == "preferred_extensions":
                results[task] = [
                    _normalize_extension(ext) for ext in af.preferred_extensions()
                ]
            elif task_key == "stable_extensions":
                results[task] = [
                    _normalize_extension(ext) for ext in af.stable_extensions()
                ]
            elif task_key == "complete_extensions":
                results[task] = [
                    _normalize_extension(ext) for ext in af.complete_extensions()
                ]
            elif task_key == "grounded_labeling":
                results[task] = af.labelings("grounded")[0]
            else:
                raise ValueError(f"Unsupported task: {task}")

        if explain:
            results["explanations"] = {}

        return results


def _normalize_extension(extension: Iterable[str]) -> list[str]:
    return sorted(extension)
