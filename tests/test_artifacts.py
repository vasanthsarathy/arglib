import json

from arglib.ai.artifacts import load_artifact_manifest, resolve_or_download


def test_load_artifact_manifest_parses_refs(tmp_path):
    manifest_path = tmp_path / "hf_artifacts.json"
    manifest_path.write_text(
        json.dumps(
            {
                "small_tagger_v1": {
                    "repo_id": "user/repo",
                    "revision": "main",
                    "subdir": "small_tagger_v1",
                }
            }
        ),
        encoding="utf-8",
    )
    refs = load_artifact_manifest(manifest_path)
    assert "small_tagger_v1" in refs
    assert refs["small_tagger_v1"].repo_id == "user/repo"


def test_resolve_or_download_prefers_existing_local_path(tmp_path):
    model_path = tmp_path / "small_tagger_v1.json"
    model_path.write_text("{}", encoding="utf-8")
    resolved = resolve_or_download(
        local_path=model_path,
        artifact_key="small_tagger_v1",
        auto_download=False,
    )
    assert resolved == model_path
