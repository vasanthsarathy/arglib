"""Publish large ArgLib artifacts to Hugging Face and write a local manifest."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class UploadSpec:
    key: str
    local_path: Path
    repo_id: str
    repo_type: str
    remote_subdir: str


def _require_token(explicit_token: str | None) -> str:
    token = explicit_token or os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "Hugging Face token is required. Pass --token or set HF_TOKEN."
        )
    return token


def _upload(api: Any, spec: UploadSpec, *, revision: str, create_private: bool) -> None:
    if not spec.local_path.exists():
        print(f"[skip] {spec.key}: missing {spec.local_path}")
        return
    api.create_repo(
        repo_id=spec.repo_id,
        repo_type=spec.repo_type,
        private=create_private,
        exist_ok=True,
    )
    if spec.local_path.is_dir():
        print(
            "[upload] "
            f"{spec.key}: folder {spec.local_path} -> "
            f"{spec.repo_id}/{spec.remote_subdir}"
        )
        api.upload_folder(
            repo_id=spec.repo_id,
            repo_type=spec.repo_type,
            folder_path=str(spec.local_path),
            path_in_repo=spec.remote_subdir,
            revision=revision,
            commit_message=f"Upload {spec.key}",
        )
        return
    remote_name = f"{spec.remote_subdir}/{spec.local_path.name}".replace("\\", "/")
    print(
        "[upload] "
        f"{spec.key}: file {spec.local_path} -> "
        f"{spec.repo_id}/{remote_name}"
    )
    api.upload_file(
        repo_id=spec.repo_id,
        repo_type=spec.repo_type,
        path_or_fileobj=str(spec.local_path),
        path_in_repo=remote_name,
        revision=revision,
        commit_message=f"Upload {spec.key}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-repo-id", required=True)
    parser.add_argument("--dataset-repo-id", default=None)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--token", default=None)
    parser.add_argument("--create-private", action="store_true")
    parser.add_argument("--small-model-path", default="models/small_tagger_v1.json")
    parser.add_argument("--neural-model-dir", default="models/neural_tagger_v1")
    parser.add_argument("--training-data-dir", default="data/training/multitask")
    parser.add_argument("--manifest-output", default="arglib/data/hf_artifacts.json")
    args = parser.parse_args()

    token = _require_token(args.token)
    try:
        from huggingface_hub import HfApi
    except Exception as exc:  # pragma: no cover - optional dep
        raise RuntimeError(
            "huggingface_hub is required. Install neural dependencies first."
        ) from exc

    api = HfApi(token=token)
    specs: list[UploadSpec] = [
        UploadSpec(
            key="small_tagger_v1",
            local_path=Path(args.small_model_path),
            repo_id=args.model_repo_id,
            repo_type="model",
            remote_subdir="small_tagger_v1",
        ),
        UploadSpec(
            key="neural_tagger_v1",
            local_path=Path(args.neural_model_dir),
            repo_id=args.model_repo_id,
            repo_type="model",
            remote_subdir="neural_tagger_v1",
        ),
    ]
    if args.dataset_repo_id:
        specs.append(
            UploadSpec(
                key="multitask_training_data",
                local_path=Path(args.training_data_dir),
                repo_id=args.dataset_repo_id,
                repo_type="dataset",
                remote_subdir="multitask_training_data",
            )
        )

    for spec in specs:
        _upload(
            api,
            spec,
            revision=args.revision,
            create_private=args.create_private,
        )

    manifest: dict[str, dict[str, str]] = {
        "small_tagger_v1": {
            "repo_id": args.model_repo_id,
            "revision": args.revision,
            "subdir": "small_tagger_v1",
        },
        "neural_tagger_v1": {
            "repo_id": args.model_repo_id,
            "revision": args.revision,
            "subdir": "neural_tagger_v1",
        },
    }
    if args.dataset_repo_id:
        manifest["multitask_training_data"] = {
            "repo_id": args.dataset_repo_id,
            "revision": args.revision,
            "subdir": "multitask_training_data",
        }

    output = Path(args.manifest_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[done] wrote manifest: {output}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
