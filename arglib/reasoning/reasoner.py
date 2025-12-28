"""Unified reasoning interface."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from arglib.core import ArgumentGraph
from arglib.reasoning.credibility import compute_credibility
from arglib.semantics.aba import enumerate_dispute_trees


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
            elif task_key == "credibility_propagation":
                credibility = compute_credibility(self.graph)
                results[task] = {
                    "initial_evidence": credibility.initial_evidence,
                    "final_scores": credibility.final_scores,
                }
            elif task_key == "aba_dispute_trees":
                aba = self.graph.metadata.get("aba_framework")
                if aba is None:
                    raise ValueError(
                        "ABA framework not found in graph.metadata['aba_framework']."
                    )
                results[task] = enumerate_dispute_trees(aba, sorted(aba.assumptions))
            else:
                raise ValueError(f"Unsupported task: {task}")

        if explain:
            results["explanations"] = {}

        return results


def _normalize_extension(extension: Iterable[str]) -> list[str]:
    return sorted(extension)
