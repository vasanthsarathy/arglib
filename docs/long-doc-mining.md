# Long-Document Mining

ArgLib provides a split -> mine -> reconcile pipeline for large inputs. The key pieces are:

- `Splitter` (for chunking),
- an `ArgumentMiner` (per-segment extraction),
- and `GraphReconciler` + `MergePolicy` (to unify graphs).

## Quick start
```python
from arglib.ai import LongDocumentMiner, SimpleArgumentMiner

miner = LongDocumentMiner(miner=SimpleArgumentMiner())
graph = miner.parse(long_text, doc_id="report-1")
```

## Custom splitting
```python
from arglib.ai import FixedWindowSplitter, LongDocumentMiner, SimpleArgumentMiner

splitter = FixedWindowSplitter(window_size=1200, overlap=200)
miner = LongDocumentMiner(miner=SimpleArgumentMiner(), splitter=splitter)
graph = miner.parse(long_text)
```

## LLM-assisted splitting
```python
from arglib.ai import LLMHook, LongDocumentMiner, NoOpLLMClient, PromptTemplate

hook = LLMHook(
    client=NoOpLLMClient(response='[{"id":"seg-1","text":"Chunk 1","start":0,"end":7}]'),
    template=PromptTemplate(
        system="Return JSON segments.",
        user="{input}",
    ),
)
miner = LongDocumentMiner(splitter_hook=hook)
graph = miner.parse(long_text)
```

## LLM-backed mining
```python
from arglib.ai import HookedArgumentMiner, LLMHook, NoOpLLMClient, PromptTemplate

hook = LLMHook(
    client=NoOpLLMClient(response='{"units": {}, "relations": [], "metadata": {}}'),
    template=PromptTemplate(system="Return graph JSON.", user="{input}"),
)
miner = HookedArgumentMiner(hook=hook)
graph = miner.parse("Some text")
```

## Async mining with sync adapters
```python
from arglib.ai import AsyncArgumentMinerAdapter, AsyncLongDocumentMiner, SimpleArgumentMiner

async_miner = AsyncArgumentMinerAdapter(SimpleArgumentMiner())
miner = AsyncLongDocumentMiner(miner=async_miner)
```

## Ollama local models
```python
from arglib.ai import HookedArgumentMiner, LLMHook, OllamaClient, PromptTemplate

hook = LLMHook(
    client=OllamaClient(model="llama3.1"),
    template=PromptTemplate(system="Return graph JSON.", user="{input}"),
)
miner = HookedArgumentMiner(hook=hook)
graph = miner.parse("Some text")
```

The Ollama client uses `ollama-python` if available, with an HTTP fallback.

## Deduplication and coreference hints
`MergePolicy` lets you control how claims are deduplicated. The default matches
lowercased text. You can add a similarity function:
```python
from arglib.ai import MergePolicy, SimpleGraphReconciler, token_jaccard_similarity

policy = MergePolicy(
    similarity_fn=token_jaccard_similarity,
    similarity_threshold=0.6,
)
reconciler = SimpleGraphReconciler(policy=policy)
```

## Output metadata
The reconciler records:
- `graph.metadata["segments"]` with segment offsets.
- `graph.metadata["merge_conflicts"]` when documents or evidence cards disagree.
- `unit.metadata["source_unit_ids"]` for per-segment provenance.
