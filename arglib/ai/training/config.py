"""Configuration objects for multitask training workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class TaskConfig:
    name: str
    enabled: bool = True
    weight: float = 1.0
    train_path: str | None = None
    dev_path: str | None = None
    test_path: str | None = None


@dataclass(frozen=True)
class TrainingConfig:
    output_dir: str = "reports/training"
    epochs: int = 3
    batch_size: int = 16
    max_steps_per_epoch: int | None = None
    seed: int = 7
    tasks: list[TaskConfig] = field(default_factory=list)

    def resolved_output_dir(self) -> Path:
        return Path(self.output_dir)

