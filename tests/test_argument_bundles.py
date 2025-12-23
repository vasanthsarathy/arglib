from arglib.core import ArgumentGraph


def test_argument_bundle_projection():
    graph = ArgumentGraph.new()
    a = graph.add_claim("A")
    b = graph.add_claim("B")
    c = graph.add_claim("C")
    d = graph.add_claim("D")

    graph.add_support(a, b)
    graph.add_support(c, d)
    graph.add_attack(b, c, weight=0.8)

    bundle1 = graph.define_argument([a, b])
    bundle2 = graph.define_argument([c, d])

    bundle_graph = graph.to_argument_graph()

    assert bundle1.id in bundle_graph.bundles
    assert bundle2.id in bundle_graph.bundles
    assert any(
        rel.src == bundle1.id and rel.dst == bundle2.id
        for rel in bundle_graph.relations
    )
