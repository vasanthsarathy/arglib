"""Minimal dispute tree generation for ABA."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from .framework import ABAFramework


@dataclass
class DisputeTreeNode:
    claim: str
    role: str  # "pro" or "con"
    children: list[DisputeTreeNode] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "claim": self.claim,
            "role": self.role,
            "children": [child.to_dict() for child in self.children],
        }


def build_dispute_tree(
    aba: ABAFramework,
    root: str,
    *,
    max_depth: int = 3,
) -> DisputeTreeNode:
    return _expand(aba, root, role="pro", max_depth=max_depth, depth=0, seen=set())


def _expand(
    aba: ABAFramework,
    claim: str,
    *,
    role: str,
    max_depth: int,
    depth: int,
    seen: set[tuple[str, str]],
) -> DisputeTreeNode:
    node = DisputeTreeNode(claim=claim, role=role)
    if depth >= max_depth:
        return node

    key = (claim, role)
    if key in seen:
        return node
    seen.add(key)

    if role == "pro":
        attackers = _attackers(aba, claim)
        node.children = [
            _expand(
                aba,
                attacker,
                role="con",
                max_depth=max_depth,
                depth=depth + 1,
                seen=seen,
            )
            for attacker in attackers
        ]
    else:
        defenders = _defenders(aba, claim)
        node.children = [
            _expand(
                aba,
                defender,
                role="pro",
                max_depth=max_depth,
                depth=depth + 1,
                seen=seen,
            )
            for defender in defenders
        ]
    return node


def _attackers(aba: ABAFramework, target: str) -> list[str]:
    attackers: set[str] = set()
    for assumption, contrary in aba.contraries.items():
        if contrary == target:
            attackers.add(assumption)
    return sorted(attackers)


def _defenders(aba: ABAFramework, target: str) -> list[str]:
    af = aba.to_dung()
    attackers = {src for src, dst in af.attacks if dst == target}
    defenders = {src for src, dst in af.attacks if dst in attackers}
    return sorted(defenders)


def enumerate_dispute_trees(
    aba: ABAFramework, targets: Iterable[str], *, max_depth: int = 3
) -> dict[str, dict[str, object]]:
    return {
        target: build_dispute_tree(aba, target, max_depth=max_depth).to_dict()
        for target in targets
    }
