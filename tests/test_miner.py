import asyncio

from arglib.ai import (
    ARGUMENT_MINING_TEMPLATE,
    AsyncArgumentMinerAdapter,
    AsyncLLMHook,
    AsyncLongDocumentMiner,
    AsyncNoOpLLMClient,
    HookedArgumentMiner,
    LLMHook,
    LongDocumentMiner,
    NoOpLLMClient,
    PromptTemplate,
    SimpleArgumentMiner,
    build_argument_miner,
    build_argument_mining_pipeline,
)


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


def test_hooked_argument_miner_parses_json_graph():
    response = (
        '{"units": {"c1": {"id": "c1", "text": "A", "type": "fact", '
        '"spans": [], "evidence": [], "evidence_ids": [], "metadata": {}}}, '
        '"relations": [], "metadata": {}}'
    )
    client = NoOpLLMClient(response=response)
    hook = LLMHook(
        client=client,
        template=PromptTemplate(system="sys", user="{input}"),
    )
    miner = HookedArgumentMiner(hook=hook)
    graph = miner.parse("ignored", doc_id="doc-3")

    assert list(graph.units.keys()) == ["c1"]


def test_argument_mining_hook_accepts_codefence_json():
    response = (
        "```json\n"
        '{"units": {"c1": {"id": "c1", "text": "Claim", "type": "fact", '
        '"spans": [], "evidence": [], "evidence_ids": [], "metadata": {}}}, '
        '"relations": [], "metadata": {}}\n'
        "```"
    )
    client = NoOpLLMClient(response=response)
    miner = build_argument_miner(client)
    graph = miner.parse("ignored", doc_id="doc-6")

    assert list(graph.units.keys()) == ["c1"]
    assert miner.hook.template == ARGUMENT_MINING_TEMPLATE


def test_long_document_miner_uses_splitter_hook():
    client = NoOpLLMClient(
        response='[{"id": "seg-1", "text": "Alpha", "start": 0, "end": 5}]'
    )
    hook = LLMHook(
        client=client,
        template=PromptTemplate(system="sys", user="{input}"),
    )
    miner = LongDocumentMiner(
        miner=SimpleArgumentMiner(),
        splitter_hook=hook,
    )
    graph = miner.parse("Alpha Beta", doc_id="doc-4")

    assert len(graph.units) == 1


def test_async_long_document_miner_uses_splitter_hook():
    async def run() -> int:
        client = AsyncNoOpLLMClient(
            response='[{"id": "seg-1", "text": "Alpha", "start": 0, "end": 5}]'
        )
        hook = AsyncLLMHook(
            client=client,
            template=PromptTemplate(system="sys", user="{input}"),
        )
        miner = AsyncLongDocumentMiner(
            miner=AsyncArgumentMinerAdapter(SimpleArgumentMiner()),
            splitter_hook=hook,
        )
        graph = await miner.parse("Alpha Beta", doc_id="doc-5")
        return len(graph.units)

    assert asyncio.run(run()) == 1


def test_argument_mining_pipeline_extracts_relations():
    class SequenceLLMClient:
        def __init__(self, responses: list[str]) -> None:
            self._responses = responses
            self._index = 0

        def complete(self, prompt: str, *, metadata=None) -> str:
            if self._index >= len(self._responses):
                return self._responses[-1]
            response = self._responses[self._index]
            self._index += 1
            return response

    responses = [
        '{"claims": [{"text": "Claim A", "type": "fact"}]}',
        '{"claims": [{"text": "Claim B", "type": "policy"}]}',
        '{"relations": [{"src": "c1", "dst": "c2", "kind": "support"}]}',
    ]
    client = SequenceLLMClient(responses)
    pipeline = build_argument_mining_pipeline(client)

    graph = pipeline.parse("Claim A.\n\nClaim B.", doc_id="doc-7")

    assert len(graph.units) == 2
    assert len(graph.relations) == 1
    relation = graph.relations[0]
    assert relation.kind == "support"
    assert relation.src in graph.units
    assert relation.dst in graph.units
