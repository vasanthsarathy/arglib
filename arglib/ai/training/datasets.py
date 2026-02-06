"""JSONL dataset and multitask dataloader helpers."""

from __future__ import annotations

import json
import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class JsonlTaskDataset:
    path: str
    split: str | None = None

    def load(self) -> list[dict[str, Any]]:
        rows = [
            json.loads(line)
            for line in Path(self.path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if self.split is None:
            return rows
        return [row for row in rows if str(row.get("split", "")) == self.split]


def batch_iter(
    rows: list[dict[str, Any]], *, batch_size: int, shuffle: bool, seed: int
) -> Iterator[list[dict[str, Any]]]:
    items = list(rows)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(items)
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


@dataclass
class MultiTaskRoundRobinLoader:
    task_rows: dict[str, list[dict[str, Any]]]
    batch_size: int = 16
    shuffle: bool = True
    seed: int = 7
    max_steps: int | None = None

    def iter_batches(self) -> Iterator[tuple[str, list[dict[str, Any]]]]:
        task_names = [name for name, rows in self.task_rows.items() if rows]
        if not task_names:
            return
        per_task_iters: dict[str, Iterator[list[dict[str, Any]]]] = {
            name: batch_iter(
                self.task_rows[name],
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                seed=self.seed + index,
            )
            for index, name in enumerate(task_names)
        }
        active = list(task_names)
        steps = 0
        while active:
            next_active: list[str] = []
            for name in active:
                iterator = per_task_iters[name]
                try:
                    batch = next(iterator)
                except StopIteration:
                    continue
                yield name, batch
                steps += 1
                if self.max_steps is not None and steps >= self.max_steps:
                    return
                next_active.append(name)
            active = next_active
