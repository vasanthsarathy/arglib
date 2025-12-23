from arglib.semantics import ABAFramework


def test_aba_framework_add_and_compute():
    aba = ABAFramework()
    aba.add_assumption("a")
    aba.add_contrary("a", "not_a")
    aba.add_rule(head="p", body=["a"])

    result = aba.compute(semantics="preferred")

    assert result["assumptions"] == ["a"]
    assert result["contraries"]["a"] == "not_a"
    assert result["rules"] == [{"head": "p", "body": ["a"]}]
    assert result["extensions"] == [["a"]]


def test_aba_rules_induce_attacks():
    aba = ABAFramework()
    aba.add_assumption("a")
    aba.add_assumption("b")
    aba.add_contrary("b", "p")
    aba.add_rule(head="p", body=["a"])

    af = aba.to_dung()

    assert ("a", "b") in af.attacks


def test_aba_multi_assumption_attack():
    aba = ABAFramework()
    aba.add_assumption("a")
    aba.add_assumption("b")
    aba.add_assumption("c")
    aba.add_contrary("c", "p")
    aba.add_rule(head="p", body=["a", "b"])

    af = aba.to_dung(max_assumption_set_size=2)

    assert ("{a&b}", "c") in af.attacks
