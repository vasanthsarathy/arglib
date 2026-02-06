"""Train transformer models for arglib multitask tagging shards."""

from __future__ import annotations

import argparse
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from arglib.ai.neural_model import require_transformers
from arglib.ai.training.registry import TASK_REGISTRY

require_transformers()

from transformers import (  # noqa: E402
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)


@dataclass(frozen=True)
class TaskSpec:
    kind: str
    text_a: str
    text_b: str | None = None
    label_key: str = "label"


_TASK_SPECS: dict[str, TaskSpec] = {
    "span_tagging": TaskSpec(kind="token", text_a="text", label_key="tags"),
    "claim_type": TaskSpec(kind="sequence", text_a="text"),
    "relation_class": TaskSpec(kind="sequence", text_a="src_text", text_b="dst_text"),
    "coref_link": TaskSpec(kind="sequence", text_a="source_text", text_b="target_text"),
    "deferred_pending": TaskSpec(kind="sequence", text_a="text"),
    "deferred_resolution": TaskSpec(
        kind="sequence", text_a="pending_text", text_b="target_text"
    ),
    "discourse_function": TaskSpec(kind="sequence", text_a="text"),
    "contradiction_nli": TaskSpec(
        kind="sequence",
        text_a="premise",
        text_b="hypothesis",
    ),
    "claim_evidence_stance": TaskSpec(
        kind="sequence", text_a="claim_text", text_b="evidence_text"
    ),
    "evidence_retrieval": TaskSpec(
        kind="sequence", text_a="claim_text", text_b="evidence_text"
    ),
}


def _set_eval_strategy(kwargs: dict[str, Any], value: str) -> None:
    signature = inspect.signature(TrainingArguments.__init__).parameters
    if "evaluation_strategy" in signature:
        kwargs["evaluation_strategy"] = value
    elif "eval_strategy" in signature:
        kwargs["eval_strategy"] = value


def _attach_tokenizer(trainer_kwargs: dict[str, Any], tokenizer: Any) -> None:
    signature = inspect.signature(Trainer.__init__).parameters
    if "tokenizer" in signature:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in signature:
        trainer_kwargs["processing_class"] = tokenizer


def _macro_f1(labels: list[int], preds: list[int], num_labels: int) -> float:
    if not labels:
        return 0.0
    total = 0.0
    pairs = list(zip(labels, preds, strict=False))
    for label in range(num_labels):
        tp = sum(1 for gold, pred in pairs if gold == label and pred == label)
        fp = sum(1 for gold, pred in pairs if gold != label and pred == label)
        fn = sum(1 for gold, pred in pairs if gold == label and pred != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        total += f1
    return total / num_labels


def _metrics_for_sequence(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    labels_list = labels.astype(int).tolist()
    preds_list = preds.astype(int).tolist()
    accuracy = (
        sum(
            1
            for gold, pred in zip(labels_list, preds_list, strict=False)
            if gold == pred
        )
        / len(labels_list)
        if labels_list
        else 0.0
    )
    macro_f1 = _macro_f1(labels_list, preds_list, int(logits.shape[1]))
    return {"accuracy": accuracy, "macro_f1": macro_f1}


def _metrics_for_token(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    gold_list: list[int] = []
    pred_list: list[int] = []
    for pred_seq, gold_seq in zip(preds, labels, strict=False):
        for pred, gold in zip(pred_seq, gold_seq, strict=False):
            if int(gold) == -100:
                continue
            gold_list.append(int(gold))
            pred_list.append(int(pred))
    accuracy = (
        sum(1 for g, p in zip(gold_list, pred_list, strict=False) if g == p)
        / len(gold_list)
        if gold_list
        else 0.0
    )
    label_count = int(logits.shape[-1]) if logits.ndim == 3 else 1
    macro_f1 = _macro_f1(gold_list, pred_list, label_count)
    return {"token_accuracy": accuracy, "token_macro_f1": macro_f1}


def _load_rows(path: Path, *, split: str, max_rows: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw)
        if str(row.get("split", "")) != split:
            continue
        rows.append(row)
        if max_rows is not None and len(rows) >= max_rows:
            break
    return rows


class SequenceDataset:
    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        tokenizer: Any,
        spec: TaskSpec,
        label_to_id: dict[str, int],
        max_length: int,
    ) -> None:
        self.items: list[dict[str, Any]] = []
        for row in rows:
            label_text = str(row.get(spec.label_key, ""))
            if label_text not in label_to_id:
                continue
            text_a = str(row.get(spec.text_a, "")).strip()
            if not text_a:
                continue
            text_b = (
                str(row.get(spec.text_b, "")).strip() if spec.text_b else None
            )
            encoded = tokenizer(
                text_a,
                text_b,
                truncation=True,
                max_length=max_length,
            )
            encoded["labels"] = label_to_id[label_text]
            self.items.append(encoded)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return dict(self.items[idx])


class TokenDataset:
    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        tokenizer: Any,
        label_to_id: dict[str, int],
        max_length: int,
    ) -> None:
        self.items: list[dict[str, Any]] = []
        for row in rows:
            text = str(row.get("text", ""))
            tags = row.get("tags", [])
            if not text or not isinstance(tags, list):
                continue
            char_labels = [
                str(item.get("tag", "O"))
                for item in tags
                if isinstance(item, dict)
            ]
            if len(char_labels) < len(text):
                char_labels.extend(["O"] * (len(text) - len(char_labels)))
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                return_offsets_mapping=True,
            )
            offsets = encoded.pop("offset_mapping")
            token_labels: list[int] = []
            for start, end in offsets:
                if end <= start:
                    token_labels.append(-100)
                    continue
                start_i = int(start)
                if start_i >= len(char_labels):
                    token_labels.append(label_to_id.get("O", 0))
                    continue
                label = char_labels[start_i]
                token_labels.append(label_to_id.get(label, label_to_id.get("O", 0)))
            encoded["labels"] = token_labels
            self.items.append(encoded)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return dict(self.items[idx])


def _build_datasets(
    task_name: str,
    *,
    tokenizer: Any,
    label_to_id: dict[str, int],
    shard: Path,
    max_length: int,
    max_train_rows: int | None,
    max_dev_rows: int | None,
    max_test_rows: int | None,
) -> tuple[Any, Any, Any]:
    spec = _TASK_SPECS[task_name]
    train_rows = _load_rows(shard, split="train", max_rows=max_train_rows)
    dev_rows = _load_rows(shard, split="dev", max_rows=max_dev_rows)
    test_rows = _load_rows(shard, split="test", max_rows=max_test_rows)
    if spec.kind == "token":
        return (
            TokenDataset(
                train_rows,
                tokenizer=tokenizer,
                label_to_id=label_to_id,
                max_length=max_length,
            ),
            TokenDataset(
                dev_rows,
                tokenizer=tokenizer,
                label_to_id=label_to_id,
                max_length=max_length,
            ),
            TokenDataset(
                test_rows,
                tokenizer=tokenizer,
                label_to_id=label_to_id,
                max_length=max_length,
            ),
        )
    return (
        SequenceDataset(
            train_rows,
            tokenizer=tokenizer,
            spec=spec,
            label_to_id=label_to_id,
            max_length=max_length,
        ),
        SequenceDataset(
            dev_rows,
            tokenizer=tokenizer,
            spec=spec,
            label_to_id=label_to_id,
            max_length=max_length,
        ),
        SequenceDataset(
            test_rows,
            tokenizer=tokenizer,
            spec=spec,
            label_to_id=label_to_id,
            max_length=max_length,
        ),
    )


def train_task(
    *,
    task_name: str,
    shard: Path,
    output_root: Path,
    base_model: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    max_length: int,
    max_train_rows: int | None,
    max_dev_rows: int | None,
    max_test_rows: int | None,
) -> dict[str, Any]:
    if task_name not in _TASK_SPECS:
        raise ValueError(f"Unsupported neural task: {task_name}")
    registry_task = TASK_REGISTRY[task_name]
    if registry_task.label_space is None:
        raise ValueError(f"Task has no label space: {task_name}")

    labels = list(registry_task.label_space)
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    spec = _TASK_SPECS[task_name]

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    train_ds, dev_ds, test_ds = _build_datasets(
        task_name,
        tokenizer=tokenizer,
        label_to_id=label_to_id,
        shard=shard,
        max_length=max_length,
        max_train_rows=max_train_rows,
        max_dev_rows=max_dev_rows,
        max_test_rows=max_test_rows,
    )
    if len(train_ds) == 0:
        raise ValueError(f"No usable train rows for {task_name}")

    if spec.kind == "token":
        model = AutoModelForTokenClassification.from_pretrained(
            base_model,
            num_labels=len(labels),
            label2id=label_to_id,
            id2label=id_to_label,
        )
        metric_fn = _metrics_for_token
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model,
            num_labels=len(labels),
            label2id=label_to_id,
            id2label=id_to_label,
        )
        metric_fn = _metrics_for_sequence

    task_out = output_root / task_name
    args_kwargs: dict[str, Any] = {
        "output_dir": str(task_out / "checkpoints"),
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "num_train_epochs": epochs,
        "learning_rate": learning_rate,
        "save_strategy": "no",
        "logging_strategy": "epoch",
        "report_to": [],
    }
    _set_eval_strategy(args_kwargs, "epoch" if len(dev_ds) else "no")
    args = TrainingArguments(**args_kwargs)

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": args,
        "train_dataset": train_ds,
        "eval_dataset": dev_ds if len(dev_ds) else None,
        "compute_metrics": metric_fn,
    }
    if spec.kind == "token":
        trainer_kwargs["data_collator"] = DataCollatorForTokenClassification(
            tokenizer=tokenizer
        )
    _attach_tokenizer(trainer_kwargs, tokenizer)
    trainer = Trainer(**trainer_kwargs)

    trainer.train()
    train_metrics = trainer.evaluate(train_ds, metric_key_prefix="train_eval")
    dev_metrics = (
        trainer.evaluate(dev_ds, metric_key_prefix="dev") if len(dev_ds) else {}
    )
    test_metrics = (
        trainer.evaluate(test_ds, metric_key_prefix="test") if len(test_ds) else {}
    )

    task_out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(task_out))
    tokenizer.save_pretrained(str(task_out))
    (task_out / "labels.json").write_text(
        json.dumps({"labels": labels, "task_type": spec.kind}, indent=2),
        encoding="utf-8",
    )
    return {
        "task": task_name,
        "task_type": spec.kind,
        "dataset": str(shard),
        "train_size": len(train_ds),
        "dev_size": len(dev_ds),
        "test_size": len(test_ds),
        "metrics": {
            "train_eval": train_metrics,
            "dev": dev_metrics,
            "test": test_metrics,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/training/multitask")
    parser.add_argument("--output-dir", default="models/neural_tagger_v1")
    parser.add_argument("--base-model", default="distilbert-base-uncased")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=[
            "span_tagging",
            "claim_type",
            "relation_class",
            "coref_link",
            "deferred_pending",
            "deferred_resolution",
            "discourse_function",
            "contradiction_nli",
            "claim_evidence_stance",
            "evidence_retrieval",
        ],
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-dev-rows", type=int, default=None)
    parser.add_argument("--max-test-rows", type=int, default=None)
    parser.add_argument(
        "--report-output",
        default="reports/neural_tagger_training_report.json",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    report_output = Path(args.report_output)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_output.parent.mkdir(parents=True, exist_ok=True)

    task_reports: list[dict[str, Any]] = []
    for task_name in args.tasks:
        shard = data_dir / f"{task_name}.jsonl"
        if not shard.exists():
            raise FileNotFoundError(f"Missing shard for {task_name}: {shard}")
        task_reports.append(
            train_task(
                task_name=task_name,
                shard=shard,
                output_root=output_dir,
                base_model=args.base_model,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                max_length=args.max_length,
                max_train_rows=args.max_train_rows,
                max_dev_rows=args.max_dev_rows,
                max_test_rows=args.max_test_rows,
            )
        )

    report = {
        "base_model": args.base_model,
        "output_dir": str(output_dir),
        "tasks": task_reports,
    }
    report_output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved report: {report_output}")


if __name__ == "__main__":
    main()
