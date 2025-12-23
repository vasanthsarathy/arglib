"""Minimal schema validation for graph payloads."""

from __future__ import annotations

from typing import Any, Dict, List


def validate_graph_dict(data: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if not isinstance(data, dict):
        return ["Graph payload must be a dictionary."]

    units = data.get("units")
    relations = data.get("relations")
    metadata = data.get("metadata")

    if not isinstance(units, dict):
        errors.append("units must be a dictionary of id -> unit.")
        units = {}
    if not isinstance(relations, list):
        errors.append("relations must be a list.")
        relations = []
    if metadata is not None and not isinstance(metadata, dict):
        errors.append("metadata must be an object when provided.")

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
        if "metadata" in unit and not isinstance(unit["metadata"], dict):
            errors.append(f"unit '{unit_id}' metadata must be an object.")

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

    return errors


def validate_graph_payload(data: Dict[str, Any]) -> None:
    errors = validate_graph_dict(data)
    if errors:
        message = "Invalid graph payload:\n" + "\n".join(f"- {error}" for error in errors)
        raise ValueError(message)
