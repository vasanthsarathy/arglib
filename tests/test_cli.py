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
