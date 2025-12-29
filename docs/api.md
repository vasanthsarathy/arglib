# API Reference

This is a lightweight overview of the public API. Full API documentation will expand as modules stabilize.

## Core
- `ArgumentGraph`: graph container and helpers (`arglib/core/graph.py`)
- `ArgumentUnit`, `Relation`, `TextSpan`, `EvidenceItem`: core data models
    - `ArgumentGraph.save(path)` writes JSON via the IO layer.
    - `ArgumentGraph.render(path, engine="graphviz")` writes DOT output.
- `EvidenceCard` and `SupportingDocument` for evidence pipelines.
- `ArgumentBundle` and `ArgumentBundleGraph` for argument-as-subgraph abstraction.
    - `ArgumentGraph.define_argument(...)` and `to_argument_graph()` project bundles.

## Semantics
- `DungAF`: Dung-style abstract argumentation framework (`arglib/semantics/dung.py`)
- `ABAFramework`: minimal ABA scaffolding (`arglib/semantics/aba/framework.py`)
    - `ABAFramework.dispute_trees()` for dispute tree generation.

## Reasoning
- `Reasoner`: unified reasoning entrypoint (`arglib/reasoning/reasoner.py`)
- `compute_credibility`: evidence + edge propagation scoring (`arglib/reasoning/credibility.py`)

## AI Evaluation
- `score_evidence` and `validate_edges` provide deterministic evaluation helpers.
- `HeuristicEvaluator` is a baseline; LLM adapters can swap in later.

## Mining
- `ArgumentMiner`, `SimpleArgumentMiner`, and `LongDocumentMiner` for extraction pipelines.
- `Splitter`, `ParagraphSplitter`, `FixedWindowSplitter` for long-document chunking.
- `MergePolicy` and `SimpleGraphReconciler` for merging per-segment graphs.
- `token_jaccard_similarity` as a simple dedup/coreference heuristic.

## I/O
- `dumps`/`loads`: JSON serialization (`arglib/io/json.py`)
- `validate_graph_payload`: minimal validation (`arglib/io/schema.py`)

## Visualization
- `to_dot`: Graphviz DOT export (`arglib/viz/graphviz.py`)
