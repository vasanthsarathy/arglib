"""Validation helpers for argument-mining chat datasets."""

from __future__ import annotations

from typing import Any

_SPLITS = {"train", "dev", "test"}
_SCENARIOS = {"summarize", "qa", "compare", "debate", "research"}
_ROLES = {"user", "assistant", "system"}
_CLAIM_TYPES = {"fact", "value", "policy", "other"}
_REL_KINDS = {"support", "attack"}
_DISCOURSE_FUNCTIONS = {
    "claim",
    "context",
    "reference",
    "explanation",
    "question",
    "meta",
}


def validate_chat_grounded_dict(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["dataset row must be a dictionary."]
    _validate_common_fields(data, errors)
    _validate_source_documents(data.get("source_documents"), errors)
    _validate_labels(data.get("labels"), errors)
    return errors


def validate_chat_exploration_dict(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["dataset row must be a dictionary."]
    _validate_common_fields(data, errors)
    if "source_documents" in data and not isinstance(data["source_documents"], list):
        errors.append("source_documents must be a list when provided.")
    _validate_labels(data.get("labels"), errors)
    return errors


def validate_chat_deferred_relevance_dict(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["dataset row must be a dictionary."]
    _validate_common_fields(data, errors)
    if "source_documents" in data and not isinstance(data["source_documents"], list):
        errors.append("source_documents must be a list when provided.")
    _validate_labels(data.get("labels"), errors, require_deferred=True)
    return errors


def validate_chat_grounded_payload(data: dict[str, Any]) -> None:
    errors = validate_chat_grounded_dict(data)
    if errors:
        raise ValueError(_format_errors("chat_grounded", errors))


def validate_chat_exploration_payload(data: dict[str, Any]) -> None:
    errors = validate_chat_exploration_dict(data)
    if errors:
        raise ValueError(_format_errors("chat_exploration", errors))


def validate_chat_deferred_relevance_payload(data: dict[str, Any]) -> None:
    errors = validate_chat_deferred_relevance_dict(data)
    if errors:
        raise ValueError(_format_errors("chat_deferred_relevance", errors))


def validate_chat_grounded_evidence_dict(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["dataset row must be a dictionary."]
    _validate_common_fields(data, errors)
    _validate_source_documents(data.get("source_documents"), errors)
    _validate_labels(data.get("labels"), errors)
    _validate_grounded_evidence_labels(data.get("labels"), errors)
    return errors


def validate_chat_grounded_evidence_payload(data: dict[str, Any]) -> None:
    errors = validate_chat_grounded_evidence_dict(data)
    if errors:
        raise ValueError(_format_errors("chat_grounded_evidence", errors))


def _validate_common_fields(data: dict[str, Any], errors: list[str]) -> None:
    row_id = data.get("id")
    split = data.get("split")
    scenario = data.get("scenario")
    conversation = data.get("conversation")
    if not isinstance(row_id, str) or not row_id.strip():
        errors.append("id must be a non-empty string.")
    if split not in _SPLITS:
        errors.append("split must be one of train/dev/test.")
    if scenario not in _SCENARIOS:
        errors.append(
            "scenario must be one of summarize/qa/compare/debate/research."
        )
    if not isinstance(conversation, list) or not conversation:
        errors.append("conversation must be a non-empty list.")
        return
    for idx, turn in enumerate(conversation):
        if not isinstance(turn, dict):
            errors.append(f"conversation[{idx}] must be an object.")
            continue
        turn_id = turn.get("turn_id")
        role = turn.get("role")
        text = turn.get("text")
        if not isinstance(turn_id, str) or not turn_id:
            errors.append(f"conversation[{idx}].turn_id must be a non-empty string.")
        if role not in _ROLES:
            errors.append(f"conversation[{idx}].role must be user/assistant/system.")
        if not isinstance(text, str) or not text.strip():
            errors.append(f"conversation[{idx}].text must be a non-empty string.")
        discourse_function = turn.get("discourse_function")
        if (
            discourse_function is not None
            and discourse_function not in _DISCOURSE_FUNCTIONS
        ):
            errors.append(
                "conversation["
                f"{idx}"
                "].discourse_function must be a valid discourse label."
            )


def _validate_source_documents(value: Any, errors: list[str]) -> None:
    if not isinstance(value, list) or not value:
        errors.append("source_documents must be a non-empty list.")
        return
    for idx, doc in enumerate(value):
        if not isinstance(doc, dict):
            errors.append(f"source_documents[{idx}] must be an object.")
            continue
        for key in ("doc_id", "title", "text"):
            if not isinstance(doc.get(key), str) or not doc.get(key, "").strip():
                errors.append(
                    f"source_documents[{idx}].{key} must be a non-empty string."
                )


def _validate_labels(
    value: Any, errors: list[str], *, require_deferred: bool = False
) -> None:
    if not isinstance(value, dict):
        errors.append("labels must be an object.")
        return
    claims = value.get("claims")
    relations = value.get("relations")
    if not isinstance(claims, list):
        errors.append("labels.claims must be a list.")
        claims = []
    if not isinstance(relations, list):
        errors.append("labels.relations must be a list.")
        relations = []

    claim_ids: set[str] = set()
    for idx, claim in enumerate(claims):
        if not isinstance(claim, dict):
            errors.append(f"labels.claims[{idx}] must be an object.")
            continue
        claim_id = claim.get("id")
        if not isinstance(claim_id, str) or not claim_id:
            errors.append(f"labels.claims[{idx}].id must be a non-empty string.")
            continue
        claim_ids.add(claim_id)
        if not isinstance(claim.get("text"), str) or not claim.get("text", "").strip():
            errors.append(f"labels.claims[{idx}].text must be a non-empty string.")
        if claim.get("type") not in _CLAIM_TYPES:
            errors.append(f"labels.claims[{idx}].type must be fact/value/policy/other.")
        if not isinstance(claim.get("turn_id"), str):
            errors.append(f"labels.claims[{idx}].turn_id must be a string.")
        source_span = claim.get("source_span")
        if source_span is not None:
            if not isinstance(source_span, dict):
                errors.append(f"labels.claims[{idx}].source_span must be an object.")
            else:
                start = source_span.get("start")
                end = source_span.get("end")
                if not isinstance(start, int) or start < 0:
                    errors.append(
                        f"labels.claims[{idx}].source_span.start must be >= 0."
                    )
                if not isinstance(end, int) or end < 0:
                    errors.append(
                        f"labels.claims[{idx}].source_span.end must be >= 0."
                    )
                if isinstance(start, int) and isinstance(end, int) and end < start:
                    errors.append(
                        f"labels.claims[{idx}].source_span.end must be >= start."
                    )

    for idx, rel in enumerate(relations):
        if not isinstance(rel, dict):
            errors.append(f"labels.relations[{idx}] must be an object.")
            continue
        src = rel.get("src")
        dst = rel.get("dst")
        kind = rel.get("kind")
        if src not in claim_ids:
            errors.append(f"labels.relations[{idx}].src references unknown claim.")
        if dst not in claim_ids:
            errors.append(f"labels.relations[{idx}].dst references unknown claim.")
        if kind not in _REL_KINDS:
            errors.append(f"labels.relations[{idx}].kind must be support or attack.")

    deferred = value.get("deferred_relevance")
    if require_deferred and not isinstance(deferred, dict):
        errors.append("labels.deferred_relevance must be an object.")
        return
    if deferred is None:
        return
    if not isinstance(deferred, dict):
        errors.append("labels.deferred_relevance must be an object when provided.")
        return
    pending_items = deferred.get("pending_items")
    resolution_links = deferred.get("resolution_links")
    if not isinstance(pending_items, list):
        errors.append("labels.deferred_relevance.pending_items must be a list.")
        pending_items = []
    if not isinstance(resolution_links, list):
        errors.append("labels.deferred_relevance.resolution_links must be a list.")
        resolution_links = []
    pending_ids: set[str] = set()
    for idx, item in enumerate(pending_items):
        if not isinstance(item, dict):
            errors.append(
                f"labels.deferred_relevance.pending_items[{idx}] must be an object."
            )
            continue
        claim_id = item.get("claim_id")
        created_turn = item.get("created_turn")
        status = item.get("status")
        if claim_id not in claim_ids:
            errors.append(
                "labels.deferred_relevance.pending_items["
                f"{idx}"
                "] claim_id must reference a known claim."
            )
        else:
            pending_ids.add(str(claim_id))
        if not isinstance(created_turn, str):
            errors.append(
                "labels.deferred_relevance.pending_items["
                f"{idx}"
                "].created_turn must be a string."
            )
        if status not in {"pending", "resolved", "discarded"}:
            errors.append(
                "labels.deferred_relevance.pending_items["
                f"{idx}"
                "].status must be pending/resolved/discarded."
            )
    for idx, link in enumerate(resolution_links):
        if not isinstance(link, dict):
            errors.append(
                f"labels.deferred_relevance.resolution_links[{idx}] must be an object."
            )
            continue
        pending_claim_id = link.get("pending_claim_id")
        target_claim_id = link.get("target_claim_id")
        resolver_turn = link.get("resolver_turn")
        if pending_claim_id not in pending_ids:
            errors.append(
                "labels.deferred_relevance.resolution_links["
                f"{idx}"
                "] pending_claim_id must reference pending_items."
            )
        if target_claim_id not in claim_ids:
            errors.append(
                "labels.deferred_relevance.resolution_links["
                f"{idx}"
                "] target_claim_id must reference known claim."
            )
        if not isinstance(resolver_turn, str):
            errors.append(
                "labels.deferred_relevance.resolution_links["
                    f"{idx}"
                    "].resolver_turn must be a string."
            )


def _validate_grounded_evidence_labels(value: Any, errors: list[str]) -> None:
    if not isinstance(value, dict):
        return
    claims = value.get("claims")
    if not isinstance(claims, list):
        return
    claim_ids = {
        str(claim.get("id"))
        for claim in claims
        if isinstance(claim, dict) and isinstance(claim.get("id"), str)
    }
    evidence_items = value.get("evidence_items")
    claim_links = value.get("claim_evidence_links")
    if not isinstance(evidence_items, list):
        errors.append("labels.evidence_items must be a list.")
        evidence_items = []
    if not isinstance(claim_links, list):
        errors.append("labels.claim_evidence_links must be a list.")
        claim_links = []

    evidence_ids: set[str] = set()
    for idx, item in enumerate(evidence_items):
        if not isinstance(item, dict):
            errors.append(f"labels.evidence_items[{idx}] must be an object.")
            continue
        evidence_id = item.get("id")
        if not isinstance(evidence_id, str) or not evidence_id:
            errors.append(
                f"labels.evidence_items[{idx}].id must be a non-empty string."
            )
            continue
        evidence_ids.add(evidence_id)
        if not isinstance(item.get("doc_id"), str) or not item.get("doc_id"):
            errors.append(
                f"labels.evidence_items[{idx}].doc_id must be a non-empty string."
            )
        if not isinstance(item.get("text"), str) or not item.get("text", "").strip():
            errors.append(
                f"labels.evidence_items[{idx}].text must be a non-empty string."
            )
        strength = item.get("strength")
        if strength is not None and not isinstance(strength, (int, float)):
            errors.append(
                "labels.evidence_items["
                f"{idx}"
                "].strength must be numeric when provided."
            )
        doc_span = item.get("doc_span")
        if doc_span is not None:
            if not isinstance(doc_span, dict):
                errors.append(
                    f"labels.evidence_items[{idx}].doc_span must be an object."
                )
            else:
                start = doc_span.get("start")
                end = doc_span.get("end")
                if not isinstance(start, int) or start < 0:
                    errors.append(
                        f"labels.evidence_items[{idx}].doc_span.start must be >= 0."
                    )
                if not isinstance(end, int) or end < 0:
                    errors.append(
                        f"labels.evidence_items[{idx}].doc_span.end must be >= 0."
                    )
                if isinstance(start, int) and isinstance(end, int) and end < start:
                    errors.append(
                        f"labels.evidence_items[{idx}].doc_span.end must be >= start."
                    )

    for idx, link in enumerate(claim_links):
        if not isinstance(link, dict):
            errors.append(f"labels.claim_evidence_links[{idx}] must be an object.")
            continue
        claim_id = link.get("claim_id")
        evidence_id = link.get("evidence_id")
        stance = link.get("stance")
        if claim_id not in claim_ids:
            errors.append(
                "labels.claim_evidence_links["
                f"{idx}"
                "].claim_id must reference known claim."
            )
        if evidence_id not in evidence_ids:
            errors.append(
                "labels.claim_evidence_links["
                f"{idx}"
                "].evidence_id must reference known evidence."
            )
        if stance not in {"support", "attack", "insufficient"}:
            errors.append(
                "labels.claim_evidence_links["
                f"{idx}"
                "].stance must be support/attack/insufficient."
            )
        confidence = link.get("confidence")
        if confidence is not None and not isinstance(confidence, (int, float)):
            errors.append(
                "labels.claim_evidence_links["
                f"{idx}"
                "].confidence must be numeric when provided."
            )


def _format_errors(name: str, errors: list[str]) -> str:
    lines = "\n".join(f"- {error}" for error in errors)
    return f"Invalid {name} payload:\n{lines}"


__all__ = [
    "validate_chat_grounded_dict",
    "validate_chat_grounded_payload",
    "validate_chat_exploration_dict",
    "validate_chat_exploration_payload",
    "validate_chat_deferred_relevance_dict",
    "validate_chat_deferred_relevance_payload",
    "validate_chat_grounded_evidence_dict",
    "validate_chat_grounded_evidence_payload",
]
