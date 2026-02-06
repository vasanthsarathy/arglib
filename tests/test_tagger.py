from arglib.ai import ClaimRelationTagger, ConversationMemory


def test_claim_relation_tagger_extracts_types_and_relations():
    text = (
        "The city has rising heat exposure. "
        "Therefore, the city should fund more parks. "
        "However, that policy is too expensive."
    )
    tagger = ClaimRelationTagger()

    result = tagger.tag(text, doc_id="doc-a")

    assert len(result.claims) == 3
    types = {claim.claim_type for claim in result.claims}
    assert "fact" in types
    assert "policy" in types
    assert len(result.relations) >= 2
    assert any(rel.kind == "support" for rel in result.relations)
    assert any(rel.kind == "attack" for rel in result.relations)


def test_conversation_memory_detects_contradiction_updates():
    memory = ConversationMemory()
    first = memory.ingest_turn("The model is reliable.", speaker="assistant")
    second = memory.ingest_turn("The model is not reliable.", speaker="assistant")

    assert first.added_claim_ids
    assert second.added_claim_ids
    assert second.contradictions
    assert any(rel["kind"] == "attack" for rel in second.added_relations)


def test_conversation_memory_merges_repeated_claims():
    memory = ConversationMemory()
    memory.ingest_turn("Solar panels reduce electricity bills.", speaker="assistant")
    update = memory.ingest_turn(
        "Solar panels reduce electricity bills.", speaker="assistant"
    )

    assert len(memory.graph.units) == 1
    assert update.merged_claim_ids


def test_conversation_memory_deferred_relevance_resolution():
    memory = ConversationMemory()
    first = memory.ingest_turn(
        "Main point stands. Side note (may seem unrelated): revenue dropped.",
        speaker="assistant",
    )
    second = memory.ingest_turn(
        "To clarify the earlier side note, this is relevant because it explains risk.",
        speaker="assistant",
    )

    assert first.pending_created
    assert second.pending_resolved
    pending_items = memory.graph.metadata["conversation"]["pending_items"]
    assert any(item.get("status") == "resolved" for item in pending_items)


def test_conversation_memory_links_anaphoric_followup_to_recent_context():
    memory = ConversationMemory()
    memory.ingest_turn(
        "The model is useful. Therefore we should deploy it. "
        "However, it is not reliable in all domains.",
        speaker="assistant",
    )
    update = memory.ingest_turn(
        "I've used it for summarizing and it is terrible.",
        speaker="assistant",
    )

    assert update.added_claim_ids
    new_id = update.added_claim_ids[0]
    assert any(
        rel["src"] == new_id and rel["kind"] in {"support", "attack"}
        for rel in update.added_relations
    )
