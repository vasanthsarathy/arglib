"""Unified reasoning interface."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from arglib.core import ArgumentGraph
from arglib.reasoning.credibility import compute_credibility
from arglib.reasoning.explain import explain_credibility


class Reasoner:
    def __init__(self, graph: ArgumentGraph) -> None:
        self.graph = graph

    def run(self, tasks: Iterable[str], explain: bool = False) -> dict[str, Any]:
        results: dict[str, Any] = {}
        last_credibility = None

        for task in tasks:
            task_key = task.lower()
            if task_key in {"credibility_propagation", "credibility"}:
                last_credibility = compute_credibility(self.graph)
                results[task] = {
                    "initial_evidence": last_credibility.initial_evidence,
                    "warrant_scores": last_credibility.warrant_scores,
                    "gate_scores": last_credibility.gate_scores,
                    "final_scores": last_credibility.final_scores,
                }
            elif task_key in {"claim_scores", "warrant_scores", "gate_scores"}:
                if last_credibility is None:
                    last_credibility = compute_credibility(self.graph)
                if task_key == "claim_scores":
                    results[task] = last_credibility.final_scores
                elif task_key == "warrant_scores":
                    results[task] = last_credibility.warrant_scores
                else:
                    results[task] = last_credibility.gate_scores
            else:
                raise ValueError(f"Unsupported task: {task}")

        if explain and last_credibility is not None:
            results["explanations"] = explain_credibility(
                self.graph, last_credibility
            )
        elif explain:
            results["explanations"] = {}

        return results
