"""Critique helpers."""

from __future__ import annotations

from typing import Dict, List

from arglib.core import ArgumentGraph


def suggest_missing_assumptions(graph: ArgumentGraph) -> List[Dict[str, str]]:
    return []