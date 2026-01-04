"""Minimal schema validation for graph payloads."""

from __future__ import annotations

from typing import Any


def validate_graph_dict(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["Graph payload must be a dictionary."]

    units = data.get("units")
    relations = data.get("relations")
    metadata = data.get("metadata")
    evidence_cards = data.get("evidence_cards", {})
    supporting_documents = data.get("supporting_documents", {})
    argument_bundles = data.get("argument_bundles", {})
    warrants = data.get("warrants", {})
    warrant_attacks = data.get("warrant_attacks", [])

    if not isinstance(units, dict):
        errors.append("units must be a dictionary of id -> unit.")
        units = {}
    if not isinstance(relations, list):
        errors.append("relations must be a list.")
        relations = []
    if metadata is not None and not isinstance(metadata, dict):
        errors.append("metadata must be an object when provided.")
    if evidence_cards is not None and not isinstance(evidence_cards, dict):
        errors.append("evidence_cards must be an object when provided.")
        evidence_cards = {}
    if supporting_documents is not None and not isinstance(supporting_documents, dict):
        errors.append("supporting_documents must be an object when provided.")
        supporting_documents = {}
    if argument_bundles is not None and not isinstance(argument_bundles, dict):
        errors.append("argument_bundles must be an object when provided.")
        argument_bundles = {}
    if warrants is not None and not isinstance(warrants, dict):
        errors.append("warrants must be an object when provided.")
        warrants = {}
    if warrant_attacks is not None and not isinstance(warrant_attacks, list):
        errors.append("warrant_attacks must be a list when provided.")
        warrant_attacks = []

    for unit_id, unit in units.items():
        if not isinstance(unit_id, str):
            errors.append("unit keys must be strings.")
            continue
        if not isinstance(unit, dict):
            errors.append(f"unit '{unit_id}' must be an object.")
            continue
        if "id" not in unit or "text" not in unit:
            errors.append(f"unit '{unit_id}' must include 'id' and 'text'.")
        if unit.get("id") != unit_id:
            errors.append(f"unit '{unit_id}' id field does not match its key.")
        if "spans" in unit and not isinstance(unit["spans"], list):
            errors.append(f"unit '{unit_id}' spans must be a list.")
        if "evidence" in unit and not isinstance(unit["evidence"], list):
            errors.append(f"unit '{unit_id}' evidence must be a list.")
        if "evidence_ids" in unit and not isinstance(unit["evidence_ids"], list):
            errors.append(f"unit '{unit_id}' evidence_ids must be a list.")
        if "evidence_min" in unit and unit["evidence_min"] is not None:
            if not isinstance(unit["evidence_min"], (int, float)):
                errors.append(f"unit '{unit_id}' evidence_min must be a number.")
        if "evidence_max" in unit and unit["evidence_max"] is not None:
            if not isinstance(unit["evidence_max"], (int, float)):
                errors.append(f"unit '{unit_id}' evidence_max must be a number.")
        if "score" in unit and unit["score"] is not None:
            if not isinstance(unit["score"], (int, float)):
                errors.append(f"unit '{unit_id}' score must be a number.")
        if "is_axiom" in unit and not isinstance(unit["is_axiom"], bool):
            errors.append(f"unit '{unit_id}' is_axiom must be a boolean.")
        if "ignore_influence" in unit and not isinstance(
            unit["ignore_influence"], bool
        ):
            errors.append(f"unit '{unit_id}' ignore_influence must be a boolean.")
        if "metadata" in unit and not isinstance(unit["metadata"], dict):
            errors.append(f"unit '{unit_id}' metadata must be an object.")
        for evidence_id in unit.get("evidence_ids", []):
            if evidence_cards and evidence_id not in evidence_cards:
                errors.append(
                    f"unit '{unit_id}' evidence_id '{evidence_id}' is not a known "
                    f"evidence card."
                )

    for card_id, card in evidence_cards.items():
        if not isinstance(card_id, str):
            errors.append("evidence_card keys must be strings.")
            continue
        if not isinstance(card, dict):
            errors.append(f"evidence_card '{card_id}' must be an object.")
            continue
        if "id" not in card or "title" not in card:
            errors.append(f"evidence_card '{card_id}' must include 'id' and 'title'.")
        if card.get("id") != card_id:
            errors.append(f"evidence_card '{card_id}' id field does not match its key.")
        supporting_doc_id = card.get("supporting_doc_id")
        if supporting_doc_id and supporting_doc_id not in supporting_documents:
            errors.append(
                f"evidence_card '{card_id}' references unknown supporting_doc_id."
            )

    for doc_id, doc in supporting_documents.items():
        if not isinstance(doc_id, str):
            errors.append("supporting_document keys must be strings.")
            continue
        if not isinstance(doc, dict):
            errors.append(f"supporting_document '{doc_id}' must be an object.")
            continue
        if "id" not in doc or "name" not in doc:
            errors.append(
                f"supporting_document '{doc_id}' must include 'id' and 'name'."
            )
        if doc.get("id") != doc_id:
            errors.append(
                f"supporting_document '{doc_id}' id field does not match its key."
            )
        if "trust" in doc and doc["trust"] is not None:
            if not isinstance(doc["trust"], (int, float)):
                errors.append(
                    f"supporting_document '{doc_id}' trust must be a number."
                )

    for warrant_id, warrant in warrants.items():
        if not isinstance(warrant_id, str):
            errors.append("warrant keys must be strings.")
            continue
        if not isinstance(warrant, dict):
            errors.append(f"warrant '{warrant_id}' must be an object.")
            continue
        if "id" not in warrant or "text" not in warrant:
            errors.append(f"warrant '{warrant_id}' must include 'id' and 'text'.")
        if warrant.get("id") != warrant_id:
            errors.append(f"warrant '{warrant_id}' id field does not match its key.")
        if "evidence_ids" in warrant and not isinstance(warrant["evidence_ids"], list):
            errors.append(f"warrant '{warrant_id}' evidence_ids must be a list.")
        if "score" in warrant and warrant["score"] is not None:
            if not isinstance(warrant["score"], (int, float)):
                errors.append(f"warrant '{warrant_id}' score must be a number.")
        if "is_axiom" in warrant and not isinstance(warrant["is_axiom"], bool):
            errors.append(f"warrant '{warrant_id}' is_axiom must be a boolean.")
        if "ignore_influence" in warrant and not isinstance(
            warrant["ignore_influence"], bool
        ):
            errors.append(f"warrant '{warrant_id}' ignore_influence must be a boolean.")
        for evidence_id in warrant.get("evidence_ids", []):
            if evidence_cards and evidence_id not in evidence_cards:
                errors.append(
                    f"warrant '{warrant_id}' evidence_id '{evidence_id}' is not a "
                    "known evidence card."
                )

    for bundle_id, bundle in argument_bundles.items():
        if not isinstance(bundle_id, str):
            errors.append("argument_bundle keys must be strings.")
            continue
        if not isinstance(bundle, dict):
            errors.append(f"argument_bundle '{bundle_id}' must be an object.")
            continue
        if bundle.get("id") != bundle_id:
            errors.append(
                f"argument_bundle '{bundle_id}' id field does not match its key."
            )
        if "units" in bundle and not isinstance(bundle["units"], list):
            errors.append(f"argument_bundle '{bundle_id}' units must be a list.")
        for unit_id in bundle.get("units", []):
            if units and unit_id not in units:
                errors.append(
                    f"argument_bundle '{bundle_id}' references unknown unit "
                    f"'{unit_id}'."
                )

    for index, relation in enumerate(relations):
        if not isinstance(relation, dict):
            errors.append(f"relation[{index}] must be an object.")
            continue
        for key in ("src", "dst", "kind"):
            if key not in relation:
                errors.append(f"relation[{index}] missing '{key}'.")
        src = relation.get("src")
        dst = relation.get("dst")
        if isinstance(src, str) and src not in units:
            errors.append(f"relation[{index}] src '{src}' is not a known unit id.")
        if isinstance(dst, str) and dst not in units:
            errors.append(f"relation[{index}] dst '{dst}' is not a known unit id.")
        if "warrant_ids" in relation and not isinstance(
            relation.get("warrant_ids", []), list
        ):
            errors.append(f"relation[{index}] warrant_ids must be a list.")
        for warrant_id in relation.get("warrant_ids", []):
            if warrants and warrant_id not in warrants:
                errors.append(
                    f"relation[{index}] warrant_id '{warrant_id}' is not a known "
                    "warrant."
                )
        gate_mode = relation.get("gate_mode")
        if gate_mode is not None and gate_mode not in {"AND", "OR"}:
            errors.append(f"relation[{index}] gate_mode must be AND or OR.")

    for index, attack in enumerate(warrant_attacks):
        if not isinstance(attack, dict):
            errors.append(f"warrant_attacks[{index}] must be an object.")
            continue
        if "src" not in attack or "warrant_id" not in attack:
            errors.append(f"warrant_attacks[{index}] missing 'src' or 'warrant_id'.")
            continue
        if isinstance(attack.get("src"), str) and attack.get("src") not in units:
            errors.append(
                f"warrant_attacks[{index}] src '{attack.get('src')}' is not a "
                "known unit id."
            )
        if isinstance(attack.get("warrant_id"), str) and attack.get(
            "warrant_id"
        ) not in warrants:
            errors.append(
                f"warrant_attacks[{index}] warrant_id '{attack.get('warrant_id')}' "
                "is not a known warrant."
            )

    return errors


def validate_graph_payload(data: dict[str, Any]) -> None:
    errors = validate_graph_dict(data)
    if errors:
        message = "Invalid graph payload:\n" + "\n".join(
            f"- {error}" for error in errors
        )
        raise ValueError(message)
