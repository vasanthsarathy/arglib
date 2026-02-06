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

## Reasoning
- `Reasoner`: unified reasoning entrypoint (`arglib/reasoning/reasoner.py`)
- `compute_credibility`: warrant-gated credibility propagation (`arglib/reasoning/credibility.py`)
- `explain_credibility`: break down evidence and gated influences (`arglib/reasoning/explain.py`)

## AI Evaluation
- `score_evidence` and `validate_edges` provide deterministic evaluation helpers.
- `HeuristicEvaluator` is a baseline; LLM adapters can swap in later.

## Critique
- `detect_patterns` and `apply_gate_actions` for flaw detection and gate invalidation.
- `analyze_warrant_fragility` to identify critical warrants per edge.

## Mining
- `ArgumentMiner`, `SimpleArgumentMiner`, and `LongDocumentMiner` for extraction pipelines.
- `ClaimRelationTagger` for fast span-based claim typing and support/attack tagging.
- `HybridClaimRelationTagger` for small-model inference with deterministic fallback.
- `HybridClaimRelationTagger` supports optional neural backend via `neural_model_dir`.
- `HybridClaimRelationTagger` supports optional Hugging Face artifact auto-download (`auto_download_artifacts`, `artifact_manifest_path`, `artifact_cache_dir`).
- `HybridClaimRelationTagger.predict_evidence_links(...)` for claim-evidence retrieval + stance inference.
- `ConversationMemory` for incremental chat-turn ingestion, claim dedup, and contradiction links.
- `Splitter`, `ParagraphSplitter`, `FixedWindowSplitter` for long-document chunking.
- `MergePolicy` and `SimpleGraphReconciler` for merging per-segment graphs.
- `token_jaccard_similarity` as a simple dedup/coreference heuristic.
- `LLMClient`/`LLMHook` and async variants for model-backed parsing or splitting.
- `OllamaClient` for local model inference.
- `AsyncArgumentMinerAdapter` to wrap sync miners for async pipelines.

## I/O
- `dumps`/`loads`: JSON serialization (`arglib/io/json.py`)
- `validate_graph_payload`: minimal validation (`arglib/io/schema.py`)
- `validate_chat_grounded_payload` and `validate_chat_exploration_payload`: dataset row validation (`arglib/io/datasets.py`)
- `validate_chat_deferred_relevance_payload`: deferred-relevance dataset validation (`arglib/io/datasets.py`)
- `validate_chat_grounded_evidence_payload`: grounded-evidence dataset validation (`arglib/io/datasets.py`)

## Visualization
- `to_dot`: Graphviz DOT export (`arglib/viz/graphviz.py`)

## Integrations
- `run_server` in `arglib/integrations/demo_ui.py` powers the local demo UI (`arglib demo-ui`).
- `scripts/publish_hf_artifacts.py` uploads heavy artifacts to Hugging Face and writes `arglib/data/hf_artifacts.json`.
- `scripts/download_hf_artifacts.py` downloads artifacts from the manifest.

## Training
- `TASK_REGISTRY`, `TaskConfig`, `TrainingConfig`, and `train_multitask` in `arglib/ai/training`.
- `JsonlTaskDataset` and `MultiTaskRoundRobinLoader` for task shard loading and round-robin batch scheduling.
- `scripts/train_neural_tagger.py` trains transformer models for sequence + token tasks:
  - `span_tagging`, `claim_type`, `relation_class`, `coref_link`
  - `deferred_pending`, `deferred_resolution`, `discourse_function`, `contradiction_nli`
  - `claim_evidence_stance`, `evidence_retrieval`
