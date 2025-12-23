from arglib.core import ArgumentGraph
from arglib.semantics import DungAF


def test_grounded_extension_simple_chain():
    af = DungAF()
    af.add_attack("a", "b")
    af.add_attack("b", "c")

    grounded = af.grounded_extension()
    assert grounded == {"a", "c"}


def test_preferred_and_stable_extensions():
    af = DungAF(arguments={"a", "b"}, attacks={("a", "b"), ("b", "a")})

    preferred = af.preferred_extensions()
    assert {"a"} in preferred
    assert {"b"} in preferred

    stable = af.stable_extensions()
    assert stable == preferred


def test_to_dung_conversion():
    graph = ArgumentGraph.new(title="Example")
    c1 = graph.add_claim("Claim 1")
    c2 = graph.add_claim("Claim 2")
    graph.add_attack(c1, c2)
    graph.add_support(c2, c1)

    af = graph.to_dung()

    assert (c1, c2) in af.attacks
    assert (c2, c1) not in af.attacks
