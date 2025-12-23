"""Assumption-based argumentation framework (minimal)."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

from arglib.semantics.dung import DungAF


@dataclass
class Rule:
    head: str
    body: list[str] = field(default_factory=list)


@dataclass
class ABAFramework:
    assumptions: set[str] = field(default_factory=set)
    contraries: dict[str, str] = field(default_factory=dict)
    rules: list[Rule] = field(default_factory=list)

    def add_assumption(self, assumption: str) -> None:
        self.assumptions.add(assumption)

    def add_contrary(self, assumption: str, contrary: str) -> None:
        self.contraries[assumption] = contrary

    def add_rule(self, head: str, body: list[str] | None = None) -> None:
        self.rules.append(Rule(head=head, body=list(body or [])))

    def _derive(self, premises: set[str]) -> set[str]:
        derived = set(premises)
        changed = True
        while changed:
            changed = False
            for rule in self.rules:
                if rule.head in derived:
                    continue
                if all(term in derived for term in rule.body):
                    derived.add(rule.head)
                    changed = True
        return derived

    def to_dung(self, max_assumption_set_size: int = 2) -> DungAF:
        af = DungAF(arguments=set(self.assumptions))
        assumption_sets = self._assumption_sets(max_assumption_set_size)
        for attacker_set in assumption_sets:
            derived = self._derive(set(attacker_set))
            for target, contrary in self.contraries.items():
                if contrary in derived:
                    attacker_id = self._format_assumption_set(attacker_set)
                    af.add_attack(attacker_id, target)
        return af

    def _assumption_sets(self, max_size: int) -> list[tuple[str, ...]]:
        if max_size < 1:
            return []
        sorted_assumptions = sorted(self.assumptions)
        sets: list[tuple[str, ...]] = []
        for size in range(1, max_size + 1):
            sets.extend(combinations(sorted_assumptions, size))
        return sets

    @staticmethod
    def _format_assumption_set(assumptions: tuple[str, ...]) -> str:
        if len(assumptions) == 1:
            return assumptions[0]
        return f"{{{'&'.join(assumptions)}}}"

    def compute(self, semantics: str = "preferred") -> dict[str, object]:
        af = self.to_dung()
        try:
            extensions = [sorted(ext) for ext in af.extensions(semantics)]
        except ValueError:
            extensions = []
        note = "Computed via a minimal assumption-contrary translation."
        return {
            "semantics": semantics,
            "assumptions": sorted(self.assumptions),
            "contraries": dict(self.contraries),
            "rules": [
                {"head": rule.head, "body": list(rule.body)} for rule in self.rules
            ],
            "extensions": extensions,
            "note": note,
        }
