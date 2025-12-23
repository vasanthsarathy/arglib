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
