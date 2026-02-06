"""Run multitask training skeleton over task JSONL shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from arglib.ai.training import (
    TASK_REGISTRY,
    TaskConfig,
    TrainingConfig,
    train_multitask,
)


def _build_task_configs(data_dir: Path) -> list[TaskConfig]:
    configs: list[TaskConfig] = []
    for task_name in TASK_REGISTRY:
        path = data_dir / f"{task_name}.jsonl"
        if not path.exists():
            continue
        configs.append(
            TaskConfig(
                name=task_name,
                enabled=True,
                weight=1.0,
                train_path=str(path),
                dev_path=str(path),
                test_path=str(path),
            )
        )
    return configs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/training/multitask")
    parser.add_argument("--output-dir", default="reports/training")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-steps-per-epoch", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    tasks = _build_task_configs(data_dir)
    if not tasks:
        raise SystemExit(
            f"No task shards found in {data_dir}. "
            "Run scripts/build_multitask_training_data.py first."
        )
    config = TrainingConfig(
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_steps_per_epoch=args.max_steps_per_epoch,
        seed=args.seed,
        tasks=tasks,
    )
    result = train_multitask(config)
    payload = {
        "steps_per_task": result.steps_per_task,
        "examples_per_task": result.examples_per_task,
        "output_report": result.output_report,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
