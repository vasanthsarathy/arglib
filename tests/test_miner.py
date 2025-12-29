from arglib.ai import LongDocumentMiner, SimpleArgumentMiner


def test_simple_argument_miner_creates_claims_with_spans():
    miner = SimpleArgumentMiner()
    graph = miner.parse("A sentence. Another one.", doc_id="doc-1")

    assert len(graph.units) == 2
    unit = next(iter(graph.units.values()))
    assert unit.spans[0].doc_id == "doc-1"
    assert unit.spans[0].start == 0


def test_long_document_miner_offsets_spans():
    miner = LongDocumentMiner(miner=SimpleArgumentMiner())
    graph = miner.parse("First.\n\nSecond.", doc_id="doc-2")

    spans = [span for unit in graph.units.values() for span in unit.spans]
    starts = sorted(span.start for span in spans)

    assert starts[0] == 0
    assert starts[1] > 0
