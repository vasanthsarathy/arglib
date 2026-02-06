"""Helpers for resolving large model artifacts from Hugging Face."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ArtifactRef:
    repo_id: str
    revision: str = "main"
    subdir: str | None = None


def _manifest_path() -> Path:
    configured = os.getenv("ARGLIB_ARTIFACTS_MANIFEST")
    if configured:
        return Path(configured)
    return Path("arglib/data/hf_artifacts.json")


def load_artifact_manifest(path: str | Path | None = None) -> dict[str, ArtifactRef]:
    manifest_path = Path(path) if path is not None else _manifest_path()
    if not manifest_path.exists():
        return {}
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    refs: dict[str, ArtifactRef] = {}
    for key, value in payload.items():
        if not isinstance(value, dict):
            continue
        repo_id = str(value.get("repo_id", "")).strip()
        if not repo_id:
            continue
        refs[key] = ArtifactRef(
            repo_id=repo_id,
            revision=str(value.get("revision", "main")),
            subdir=str(value["subdir"]) if "subdir" in value else None,
        )
    return refs


def download_artifact(
    artifact_key: str,
    *,
    cache_dir: str | Path | None = None,
    manifest_path: str | Path | None = None,
    local_only: bool = False,
    token: str | None = None,
) -> Path:
    refs = load_artifact_manifest(manifest_path)
    if artifact_key not in refs:
        raise KeyError(f"Artifact not found in manifest: {artifact_key}")
    ref = refs[artifact_key]
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError(
            "huggingface_hub is required for artifact download. "
            "Install neural dependencies or add huggingface-hub."
        ) from exc

    hub_kwargs: dict[str, Any] = {
        "repo_id": ref.repo_id,
        "revision": ref.revision,
        "local_files_only": local_only,
        "token": token or os.getenv("HF_TOKEN"),
    }
    if cache_dir:
        hub_kwargs["cache_dir"] = str(cache_dir)
    if ref.subdir:
        hub_kwargs["allow_patterns"] = [f"{ref.subdir}/**"]
    snapshot_root = Path(snapshot_download(**hub_kwargs))
    return snapshot_root / ref.subdir if ref.subdir else snapshot_root


def resolve_or_download(
    *,
    local_path: str | Path | None,
    artifact_key: str,
    auto_download: bool = False,
    cache_dir: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> Path | None:
    if local_path:
        path = Path(local_path)
        if path.exists():
            return path
    if not auto_download:
        return None
    return download_artifact(
        artifact_key,
        cache_dir=cache_dir,
        manifest_path=manifest_path,
    )


__all__ = [
    "ArtifactRef",
    "download_artifact",
    "load_artifact_manifest",
    "resolve_or_download",
]
