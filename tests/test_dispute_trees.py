from arglib.semantics.aba import ABAFramework, build_dispute_tree


def test_dispute_tree_simple_attack():
    aba = ABAFramework()
    aba.add_assumption("a")
    aba.add_assumption("b")
    aba.add_contrary("b", "a")

    tree = build_dispute_tree(aba, "a", max_depth=2)

    assert tree.claim == "a"
    assert tree.role == "pro"
    assert tree.children
    assert tree.children[0].claim == "b"
    assert tree.children[0].role == "con"
