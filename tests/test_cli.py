import importlib
import json

from arglib.cli.main import main
from arglib.core import ArgumentGraph
from arglib.io import save


def test_cli_dot(tmp_path, capsys):
    graph = ArgumentGraph.new()
    a = graph.add_claim("A")
    b = graph.add_claim("B")
    graph.add_attack(a, b)
    path = tmp_path / "graph.json"
    save(path, graph)

    exit_code = main(["dot", str(path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "digraph ArgumentGraph" in captured.out


def test_cli_diagnostics(tmp_path, capsys):
    graph = ArgumentGraph.new()
    graph.add_claim("A")
    path = tmp_path / "graph.json"
    save(path, graph)

    exit_code = main(["diagnostics", str(path), "--validate"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["node_count"] == 1


def test_cli_validate(tmp_path, capsys):
    graph = ArgumentGraph.new()
    graph.add_claim("A")
    path = tmp_path / "graph.json"
    save(path, graph)

    exit_code = main(["validate", str(path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.strip() == "OK"


def test_cli_demo_ui_invokes_server(monkeypatch):
    cli_module = importlib.import_module("arglib.cli.main")
    called = {}

    def _fake_run_server(
        *,
        host,
        port,
        model_path,
        auto_download_artifacts,
        artifact_manifest_path,
        artifact_cache_dir,
    ):
        called["host"] = host
        called["port"] = port
        called["model_path"] = model_path
        called["auto_download_artifacts"] = auto_download_artifacts
        called["artifact_manifest_path"] = artifact_manifest_path
        called["artifact_cache_dir"] = artifact_cache_dir

    monkeypatch.setattr(cli_module, "run_server", _fake_run_server)
    exit_code = cli_module.main(["demo-ui", "--host", "127.0.0.1", "--port", "9999"])

    assert exit_code == 0
    assert called["host"] == "127.0.0.1"
    assert called["port"] == 9999
    assert called["auto_download_artifacts"] is False
