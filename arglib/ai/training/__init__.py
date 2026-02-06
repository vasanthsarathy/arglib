"""Training scaffolding for multitask argument analysis models."""

from .config import TaskConfig, TrainingConfig
from .datasets import JsonlTaskDataset, MultiTaskRoundRobinLoader
from .registry import TASK_REGISTRY, TaskSpec
from .trainer import TrainResult, train_multitask

__all__ = [
    "JsonlTaskDataset",
    "MultiTaskRoundRobinLoader",
    "TASK_REGISTRY",
    "TaskConfig",
    "TaskSpec",
    "TrainResult",
    "TrainingConfig",
    "train_multitask",
]

