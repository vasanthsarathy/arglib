from pathlib import Path

from arglib.ai.training import TaskConfig, TrainingConfig, train_multitask


def test_train_multitask_skeleton_runs(tmp_path: Path):
    shard = tmp_path / "claim_type.jsonl"
    shard.write_text(
        '{"split":"train","text":"A","label":"fact"}\n'
        '{"split":"train","text":"B","label":"value"}\n'
        '{"split":"dev","text":"C","label":"policy"}\n',
        encoding="utf-8",
    )
    config = TrainingConfig(
        output_dir=str(tmp_path / "reports"),
        epochs=2,
        batch_size=1,
        tasks=[
            TaskConfig(
                name="claim_type",
                train_path=str(shard),
                dev_path=str(shard),
                test_path=str(shard),
            )
        ],
    )
    result = train_multitask(config)
    assert result.examples_per_task["claim_type"] == 2
    assert result.steps_per_task["claim_type"] == 4
