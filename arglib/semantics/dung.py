"""Dung-style abstract argumentation frameworks."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import chain, combinations
from typing import Dict, Iterable, List, Set, Tuple


@dataclass
class DungAF:
    arguments: Set[str] = field(default_factory=set)
    attacks: Set[Tuple[str, str]] = field(default_factory=set)

    def add_argument(self, arg: str) -> None:
        self.arguments.add(arg)

    def add_attack(self, attacker: str, target: str) -> None:
        self.arguments.update([attacker, target])
        self.attacks.add((attacker, target))

    def attackers_of(self, arg: str) -> Set[str]:
        return {src for src, dst in self.attacks if dst == arg}

    def attacks_from(self, arg: str) -> Set[str]:
        return {dst for src, dst in self.attacks if src == arg}

    def conflict_free(self, extension: Iterable[str]) -> bool:
        ext = set(extension)
        return all((a, b) not in self.attacks for a in ext for b in ext)

    def defends(self, extension: Iterable[str], arg: str) -> bool:
        ext = set(extension)
        attackers = self.attackers_of(arg)
        if not attackers:
            return True
        defended_attackers = {
            attacker
            for attacker in attackers
            if any((defender, attacker) in self.attacks for defender in ext)
        }
        return attackers == defended_attackers

    def admissible_sets(self) -> List[Set[str]]:
        admissible: List[Set[str]] = []
        for ext in self._powerset(self.arguments):
            if self.conflict_free(ext) and all(self.defends(ext, arg) for arg in ext):
                admissible.append(set(ext))
        return admissible

    def complete_extensions(self) -> List[Set[str]]:
        complete: List[Set[str]] = []
        for ext in self.admissible_sets():
            defended = {arg for arg in self.arguments if self.defends(ext, arg)}
            if defended == ext:
                complete.append(ext)
        return complete

    def grounded_extension(self) -> Set[str]:
        complete = self.complete_extensions()
        if not complete:
            return set()
        return set.intersection(*complete)

    def preferred_extensions(self) -> List[Set[str]]:
        admissible = self.admissible_sets()
        preferred: List[Set[str]] = []
        for candidate in admissible:
            if not any(candidate < other for other in admissible):
                preferred.append(candidate)
        return preferred

    def stable_extensions(self) -> List[Set[str]]:
        stable: List[Set[str]] = []
        for ext in self._powerset(self.arguments):
            ext_set = set(ext)
            if not self.conflict_free(ext_set):
                continue
            attacked = set(chain.from_iterable(self.attacks_from(arg) for arg in ext_set))
            if attacked == self.arguments - ext_set:
                stable.append(ext_set)
        return stable

    def extensions(self, semantics: str) -> List[Set[str]]:
        semantics = semantics.lower()
        if semantics == "grounded":
            return [self.grounded_extension()]
        if semantics == "preferred":
            return self.preferred_extensions()
        if semantics == "stable":
            return self.stable_extensions()
        if semantics == "complete":
            return self.complete_extensions()
        raise ValueError(f"Unsupported semantics: {semantics}")

    def labelings(self, semantics: str = "grounded") -> List[Dict[str, str]]:
        semantics = semantics.lower()
        if semantics != "grounded":
            raise ValueError("Only grounded labeling is implemented.")

        labels: Dict[str, str] = {arg: "undec" for arg in self.arguments}
        changed = True
        while changed:
            changed = False
            for arg in sorted(self.arguments):
                if labels[arg] != "undec":
                    continue
                attackers = self.attackers_of(arg)
                if not attackers or all(labels[attacker] == "out" for attacker in attackers):
                    labels[arg] = "in"
                    changed = True
            for arg in sorted(self.arguments):
                if labels[arg] != "undec":
                    continue
                if any(labels[attacker] == "in" for attacker in self.attackers_of(arg)):
                    labels[arg] = "out"
                    changed = True
        return [labels]

    @staticmethod
    def _powerset(items: Iterable[str]) -> Iterable[Tuple[str, ...]]:
        items = list(items)
        return chain.from_iterable(combinations(items, r) for r in range(len(items) + 1))
