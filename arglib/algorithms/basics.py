"""Basic graph algorithms for argument graphs."""

from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, List, Sequence, Set, Tuple


def build_edges(
    relations: Iterable[Tuple[str, str, str]],
    kinds: Sequence[str] | None = None,
) -> List[Tuple[str, str]]:
    edges: List[Tuple[str, str]] = []
    for src, dst, kind in relations:
        if kinds is None or kind in kinds:
            edges.append((src, dst))
    return edges


def find_cycles(nodes: Iterable[str], edges: Iterable[Tuple[str, str]]) -> List[List[str]]:
    adjacency: Dict[str, List[str]] = {node: [] for node in nodes}
    for src, dst in edges:
        adjacency.setdefault(src, []).append(dst)
        adjacency.setdefault(dst, [])

    seen: Set[str] = set()
    stack: List[str] = []
    on_stack: Set[str] = set()
    cycles: Set[Tuple[str, ...]] = set()

    def record_cycle(start_index: int) -> None:
        cycle = stack[start_index:]
        if not cycle:
            return
        min_node = min(cycle)
        while cycle[0] != min_node:
            cycle = cycle[1:] + cycle[:1]
        cycles.add(tuple(cycle))

    def dfs(node: str) -> None:
        seen.add(node)
        stack.append(node)
        on_stack.add(node)
        for neighbor in adjacency.get(node, []):
            if neighbor not in seen:
                dfs(neighbor)
            elif neighbor in on_stack:
                record_cycle(stack.index(neighbor))
        stack.pop()
        on_stack.remove(node)

    for node in adjacency:
        if node not in seen:
            dfs(node)

    return [list(cycle) for cycle in sorted(cycles)]


def weakly_connected_components(
    nodes: Iterable[str],
    edges: Iterable[Tuple[str, str]],
) -> List[List[str]]:
    adjacency: Dict[str, Set[str]] = {node: set() for node in nodes}
    for src, dst in edges:
        adjacency.setdefault(src, set()).add(dst)
        adjacency.setdefault(dst, set()).add(src)

    remaining = set(adjacency.keys())
    components: List[List[str]] = []
    while remaining:
        start = remaining.pop()
        queue = deque([start])
        component = {start}
        while queue:
            node = queue.popleft()
            for neighbor in adjacency.get(node, set()):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)
        components.append(sorted(component))
    return components


def in_out_degree(nodes: Iterable[str], edges: Iterable[Tuple[str, str]]) -> Dict[str, Dict[str, int]]:
    degrees = {node: {"in": 0, "out": 0} for node in nodes}
    for src, dst in edges:
        degrees.setdefault(src, {"in": 0, "out": 0})
        degrees.setdefault(dst, {"in": 0, "out": 0})
        degrees[src]["out"] += 1
        degrees[dst]["in"] += 1
    return degrees


def strongly_connected_components(
    nodes: Iterable[str],
    edges: Iterable[Tuple[str, str]],
) -> List[List[str]]:
    adjacency: Dict[str, List[str]] = {node: [] for node in nodes}
    for src, dst in edges:
        adjacency.setdefault(src, []).append(dst)
        adjacency.setdefault(dst, [])

    index = 0
    index_map: Dict[str, int] = {}
    lowlink: Dict[str, int] = {}
    stack: List[str] = []
    on_stack: Set[str] = set()
    components: List[List[str]] = []

    def visit(node: str) -> None:
        nonlocal index
        index_map[node] = index
        lowlink[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)

        for neighbor in adjacency.get(node, []):
            if neighbor not in index_map:
                visit(neighbor)
                lowlink[node] = min(lowlink[node], lowlink[neighbor])
            elif neighbor in on_stack:
                lowlink[node] = min(lowlink[node], index_map[neighbor])

        if lowlink[node] == index_map[node]:
            component: List[str] = []
            while True:
                member = stack.pop()
                on_stack.remove(member)
                component.append(member)
                if member == node:
                    break
            components.append(sorted(component))

    for node in adjacency:
        if node not in index_map:
            visit(node)

    return sorted(components, key=lambda comp: (len(comp), comp))


def reachability_map(
    nodes: Iterable[str],
    edges: Iterable[Tuple[str, str]],
) -> Dict[str, Set[str]]:
    adjacency: Dict[str, Set[str]] = {node: set() for node in nodes}
    for src, dst in edges:
        adjacency.setdefault(src, set()).add(dst)
        adjacency.setdefault(dst, set())

    reachability: Dict[str, Set[str]] = {}
    for start in adjacency:
        visited: Set[str] = set()
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in adjacency.get(node, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        reachability[start] = visited
    return reachability
