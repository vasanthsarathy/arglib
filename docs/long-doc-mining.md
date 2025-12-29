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
