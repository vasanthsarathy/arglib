"""Unified reasoning interface."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from arglib.core import ArgumentGraph
from arglib.reasoning.credibility import compute_credibility


class Reasoner:
    def __init__(self, graph: ArgumentGraph) -> None:
        self.graph = graph

    def run(self, tasks: Iterable[str], explain: bool = False) -> dict[str, Any]:
        results: dict[str, Any] = {}

        for task in tasks:
            task_key = task.lower()
            if task_key in {"credibility_propagation", "credibility"}:
                credibility = compute_credibility(self.graph)
                results[task] = {
                    "initial_evidence": credibility.initial_evidence,
                    "warrant_scores": credibility.warrant_scores,
                    "gate_scores": credibility.gate_scores,
                    "final_scores": credibility.final_scores,
                }
            elif task_key in {"claim_scores", "warrant_scores", "gate_scores"}:
                credibility = compute_credibility(self.graph)
                if task_key == "claim_scores":
                    results[task] = credibility.final_scores
                elif task_key == "warrant_scores":
                    results[task] = credibility.warrant_scores
                else:
                    results[task] = credibility.gate_scores
            else:
                raise ValueError(f"Unsupported task: {task}")

        if explain:
            results["explanations"] = {}

        return results
