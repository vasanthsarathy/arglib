from arglib.core import ArgumentGraph, EvidenceCard, SupportingDocument, TextSpan
from arglib.io import dumps, loads


def test_graph_roundtrip():
    graph = ArgumentGraph.new(title="Example")
    c1 = graph.add_claim("Green spaces reduce urban heat.", type="fact")
    c2 = graph.add_claim("Cities should fund parks.", type="policy")
    graph.add_support(c1, c2, weight=0.7, rationale="Cooling improves health")

    span = TextSpan(
        doc_id="paper.pdf",
        start=0,
        end=10,
        text="...",
        page=3,
        modality="pdf",
    )
    graph.attach_evidence(
        c1,
        evidence_id="e1",
        source=span,
        stance="supports",
        strength=0.9,
    )
    graph.add_supporting_document(
        SupportingDocument(id="doc1", name="Doc", type="pdf", url="file://doc1")
    )
    graph.add_evidence_card(
        EvidenceCard(
            id="e2",
            title="Card",
            supporting_doc_id="doc1",
            excerpt="...",
            confidence=0.8,
        )
    )
    graph.attach_evidence_card(c2, "e2")

    payload = dumps(graph)
    restored = loads(payload)

    assert restored.metadata["title"] == "Example"
    assert restored.units[c1].text == "Green spaces reduce urban heat."
    assert restored.relations[0].kind == "support"
    assert restored.units[c1].evidence[0].id == "e1"
    assert restored.evidence_cards["e2"].confidence == 0.8
