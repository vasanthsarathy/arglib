"""Download ArgLib artifacts defined in a Hugging Face manifest."""

from __future__ import annotations

import argparse
import json
from typing import Any

from arglib.ai.artifacts import download_artifact, load_artifact_manifest


def _materialize_key(
    key: str,
    *,
    cache_dir: str | None,
    manifest_path: str | None,
    local_only: bool,
) -> dict[str, Any]:
    source = download_artifact(
        key,
        cache_dir=cache_dir,
        manifest_path=manifest_path,
        local_only=local_only,
    )
    refs = load_artifact_manifest(manifest_path)
    ref = refs[key]
    return {
        "key": key,
        "repo_id": ref.repo_id,
        "revision": ref.revision,
        "source": str(source),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-path", default="arglib/data/hf_artifacts.json")
    parser.add_argument(
        "--keys",
        nargs="+",
        default=["small_tagger_v1", "neural_tagger_v1"],
    )
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--local-only", action="store_true")
    args = parser.parse_args()

    manifest = load_artifact_manifest(args.manifest_path)
    missing = [key for key in args.keys if key not in manifest]
    if missing:
        raise KeyError(f"Missing keys in manifest: {missing}")

    report = []
    for key in args.keys:
        item = _materialize_key(
            key,
            cache_dir=args.cache_dir,
            manifest_path=args.manifest_path,
            local_only=args.local_only,
        )
        report.append(item)
        print(f"[downloaded] {key} -> {item['source']}")

    print(json.dumps({"artifacts": report}, indent=2))


if __name__ == "__main__":
    main()
