from arglib.core import ArgumentGraph


def test_diagnostics_cycles_and_components():
    graph = ArgumentGraph.new()
    a = graph.add_claim("A")
    b = graph.add_claim("B")
    c = graph.add_claim("C")
    d = graph.add_claim("D")

    graph.add_attack(a, b)
    graph.add_attack(b, a)
    graph.add_support(c, a)

    diagnostics = graph.diagnostics()

    assert diagnostics["node_count"] == 4
    assert diagnostics["relation_count"] == 3
    assert diagnostics["attack_edge_count"] == 2
    assert diagnostics["support_edge_count"] == 1
    assert diagnostics["cycle_count"] == 1
    assert diagnostics["component_count"] == 2
    assert diagnostics["scc_count"] == 3
    assert diagnostics["isolated_units"] == [d]
    assert diagnostics["reachability"][c] == [a, b]
    assert diagnostics["reachability"][d] == []
    assert diagnostics["degree_summary"]["max_in"] == 2
    assert diagnostics["degree_summary"]["max_out"] == 1


def test_critique_stub():
    graph = ArgumentGraph.new()
    graph.add_claim("A")

    critique = graph.critique()

    assert critique["missing_assumptions"] == []
