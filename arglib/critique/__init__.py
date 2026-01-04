"""Critique and diagnostics."""

from .assumptions import suggest_missing_assumptions
from .fragility import WarrantFragility, analyze_warrant_fragility
from .patterns import PatternMatch, apply_gate_actions, detect_patterns

__all__ = [
    "PatternMatch",
    "WarrantFragility",
    "analyze_warrant_fragility",
    "apply_gate_actions",
    "detect_patterns",
    "suggest_missing_assumptions",
]
