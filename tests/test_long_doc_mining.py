from arglib.ai import (
    FixedWindowSplitter,
    MergePolicy,
    ParagraphSplitter,
    SimpleGraphReconciler,
    token_jaccard_similarity,
)
from arglib.core import ArgumentGraph


def test_paragraph_splitter_offsets():
    text = "Alpha\n\nBeta\n\nGamma"
    splitter = ParagraphSplitter()
    segments = splitter.split(text)

    assert [seg.text for seg in segments] == ["Alpha", "Beta", "Gamma"]
    assert segments[0].start == 0
    assert segments[0].end == len("Alpha")
    assert segments[1].text == "Beta"


def test_fixed_window_splitter_overlap():
    text = "abcdefghij"
    splitter = FixedWindowSplitter(window_size=4, overlap=1)
    segments = splitter.split(text)

    assert [seg.text for seg in segments] == ["abcd", "defg", "ghij"]
    assert [seg.start for seg in segments] == [0, 3, 6]


def test_simple_graph_reconciler_merges_units_and_edges():
    g1 = ArgumentGraph.new()
    a1 = g1.add_claim("A")
    b1 = g1.add_claim("B")
    g1.add_support(a1, b1, weight=0.6)

    g2 = ArgumentGraph.new()
    a2 = g2.add_claim("A")
    b2 = g2.add_claim("B")
    c2 = g2.add_claim("C")
    g2.add_support(a2, b2, weight=0.5)
    g2.add_attack(a2, c2, weight=-0.4)

    splitter = ParagraphSplitter()
    segments = splitter.split("A\n\nB")

    reconciler = SimpleGraphReconciler()
    result = reconciler.reconcile(segments, [g1, g2])

    assert len(result.graph.units) == 3
    merged_segments = [
        segments for segments in result.unit_segments.values() if len(segments) == 2
    ]
    assert merged_segments == [["seg-1", "seg-2"], ["seg-1", "seg-2"]]

    relation_keys = {
        (rel.src, rel.dst, rel.kind): rel for rel in result.graph.relations
    }
    support_key = next(key for key in relation_keys if key[2] == "support")
    assert relation_keys[support_key].weight == 1.0
    assert len(result.relation_sources[support_key]) == 2


def test_similarity_merge_policy():
    g1 = ArgumentGraph.new()
    a1 = g1.add_claim("Cats are good.")

    g2 = ArgumentGraph.new()
    a2 = g2.add_claim("Cats are very good.")

    segments = ParagraphSplitter().split("Cats are good.\n\nCats are very good.")
    policy = MergePolicy(
        similarity_fn=token_jaccard_similarity,
        similarity_threshold=0.6,
    )
    reconciler = SimpleGraphReconciler(policy=policy)
    result = reconciler.reconcile(segments, [g1, g2])

    assert len(result.graph.units) == 1
    merged_unit = next(iter(result.graph.units.values()))
    assert sorted(merged_unit.metadata["source_unit_ids"]) == sorted([a1, a2])
