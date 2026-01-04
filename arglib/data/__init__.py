"""Data assets."""

from __future__ import annotations

from pathlib import Path


def get_patterns_bank_path() -> Path | None:
    path = Path(__file__).with_name("patterns_bank.yaml")
    if path.exists():
        return path
    return None


__all__ = ["get_patterns_bank_path"]
