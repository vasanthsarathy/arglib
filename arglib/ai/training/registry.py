"""Task registry for multitask argument analysis training."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    name: str
    description: str
    label_space: tuple[str, ...] | None = None


TASK_REGISTRY: dict[str, TaskSpec] = {
    "span_tagging": TaskSpec(
        name="span_tagging",
        description="BIO-style claim span tagging per turn text.",
        label_space=("B-CLAIM", "I-CLAIM", "O"),
    ),
    "claim_type": TaskSpec(
        name="claim_type",
        description="Claim type classification: fact/value/policy/other.",
        label_space=("fact", "value", "policy", "other"),
    ),
    "relation_class": TaskSpec(
        name="relation_class",
        description="Pairwise relation classification: support/attack/none.",
        label_space=("support", "attack", "none"),
    ),
    "coref_link": TaskSpec(
        name="coref_link",
        description="Cross-turn reference linking for new claim to prior claim.",
        label_space=("yes", "no"),
    ),
    "deferred_pending": TaskSpec(
        name="deferred_pending",
        description="Pending relevance detection at claim level.",
        label_space=("pending", "not_pending"),
    ),
    "deferred_resolution": TaskSpec(
        name="deferred_resolution",
        description="Resolution link prediction for pending claims.",
        label_space=("resolve", "none"),
    ),
    "discourse_function": TaskSpec(
        name="discourse_function",
        description="Turn-level discourse function labeling.",
        label_space=(
            "claim",
            "context",
            "reference",
            "explanation",
            "question",
            "meta",
        ),
    ),
    "contradiction_nli": TaskSpec(
        name="contradiction_nli",
        description="Pairwise contradiction entailment signal.",
        label_space=("entails", "contradicts", "neutral"),
    ),
    "claim_evidence_stance": TaskSpec(
        name="claim_evidence_stance",
        description="Claim to evidence stance: support/attack/insufficient.",
        label_space=("support", "attack", "insufficient"),
    ),
    "evidence_retrieval": TaskSpec(
        name="evidence_retrieval",
        description="Binary claim-evidence linkage for retrieval.",
        label_space=("yes", "no"),
    ),
}
