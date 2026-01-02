"""Credibility propagation for argument graphs."""

from __future__ import annotations

from dataclasses import dataclass, field

from arglib.core import ArgumentGraph

from .warrant_gated import WarrantGatedConfig, compute_warrant_gated_scores


@dataclass
class CredibilityResult:
    initial_evidence: dict[str, float]
    iterations: list[dict[str, float]]
    final_scores: dict[str, float]
    warrant_scores: dict[str, float] = field(default_factory=dict)
    gate_scores: dict[str, float] = field(default_factory=dict)


def compute_credibility(
    graph: ArgumentGraph,
    *,
    lambda_: float = 0.5,
    max_iterations: int = 100,
    epsilon: float = 1e-6,
    evidence_min: float = 0.0,
    evidence_max: float = 1.0,
    initial_scores: dict[str, float] | None = None,
) -> CredibilityResult:
    config = WarrantGatedConfig(
        alpha=lambda_,
        beta=1.0,
        max_iterations=max_iterations,
        epsilon=epsilon,
        gate_required=True,
    )
    result = compute_warrant_gated_scores(graph, config=config)
    return CredibilityResult(
        initial_evidence=result.evidence_support,
        iterations=result.claim_iterations,
        final_scores=result.final_claim_scores,
        warrant_scores=result.final_warrant_scores,
        gate_scores=result.gate_scores,
    )
