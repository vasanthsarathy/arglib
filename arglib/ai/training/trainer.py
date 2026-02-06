"""Multitask training skeleton with round-robin dataloading."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import Any

from .config import TrainingConfig
from .datasets import JsonlTaskDataset, MultiTaskRoundRobinLoader


@dataclass
class TrainResult:
    steps_per_task: dict[str, int]
    examples_per_task: dict[str, int]
    output_report: str


def train_multitask(config: TrainingConfig) -> TrainResult:
    task_rows: dict[str, list[dict[str, Any]]] = {}
    for task in config.tasks:
        if not task.enabled or not task.train_path:
            continue
        rows = JsonlTaskDataset(path=task.train_path, split="train").load()
        task_rows[task.name] = rows

    steps = Counter()
    examples = {name: len(rows) for name, rows in task_rows.items()}
    for epoch in range(config.epochs):
        loader = MultiTaskRoundRobinLoader(
            task_rows=task_rows,
            batch_size=config.batch_size,
            shuffle=True,
            seed=config.seed + epoch,
            max_steps=config.max_steps_per_epoch,
        )
        for task_name, _batch in loader.iter_batches():
            steps[task_name] += 1

    report = {
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "seed": config.seed,
        "examples_per_task": examples,
        "steps_per_task": dict(steps),
        "note": "Skeleton trainer only. Plug in model forward/loss/backprop.",
    }
    output_dir = config.resolved_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "multitask_train_report.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return TrainResult(
        steps_per_task=dict(steps),
        examples_per_task=examples,
        output_report=str(output_path),
    )
