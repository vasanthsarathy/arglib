"""Assumption-based argumentation framework (minimal)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set

from arglib.semantics.dung import DungAF


@dataclass
class Rule:
    head: str
    body: List[str] = field(default_factory=list)


@dataclass
class ABAFramework:
    assumptions: Set[str] = field(default_factory=set)
    contraries: Dict[str, str] = field(default_factory=dict)
    rules: List[Rule] = field(default_factory=list)

    def add_assumption(self, assumption: str) -> None:
        self.assumptions.add(assumption)

    def add_contrary(self, assumption: str, contrary: str) -> None:
        self.contraries[assumption] = contrary

    def add_rule(self, head: str, body: List[str] | None = None) -> None:
        self.rules.append(Rule(head=head, body=list(body or [])))

    def to_dung(self) -> DungAF:
        af = DungAF(arguments=set(self.assumptions))
        for assumption, contrary in self.contraries.items():
            if contrary in self.assumptions:
                af.add_attack(assumption, contrary)
        return af

    def compute(self, semantics: str = "preferred") -> Dict[str, object]:
        af = self.to_dung()
        try:
            extensions = [sorted(ext) for ext in af.extensions(semantics)]
        except ValueError:
            extensions = []
        return {
            "semantics": semantics,
            "assumptions": sorted(self.assumptions),
            "contraries": dict(self.contraries),
            "rules": [{"head": rule.head, "body": list(rule.body)} for rule in self.rules],
            "extensions": extensions,
            "note": "Computed via a minimal assumption-contrary translation (rules are not yet used).",
        }
