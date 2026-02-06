# ArgLib

[![PyPI](https://img.shields.io/pypi/v/arglib)](https://pypi.org/project/arglib/)
[![Python](https://img.shields.io/pypi/pyversions/arglib)](https://pypi.org/project/arglib/)
[![CI](https://github.com/vasanthsarathy/arglib/actions/workflows/ci.yml/badge.svg)](https://github.com/vasanthsarathy/arglib/actions/workflows/ci.yml)
[![Docs](https://github.com/vasanthsarathy/arglib/actions/workflows/docs.yml/badge.svg)](https://vasanthsarathy.github.io/arglib/)
[![License](https://img.shields.io/github/license/vasanthsarathy/arglib)](LICENSE)

ArgLib is a batteries-included Python library for creating, importing, analyzing, and reasoning over argument graphs derived from text and multimodal evidence.

## Highlights
- Canonical `ArgumentGraph` model with provenance-aware nodes and relations.
- Warrant-gated scoring with claim, warrant, and gate scores.
- Axiom flags to seed manual scores with optional influence locking.
- Diagnostics for cycles, components, reachability, and degree stats.
- JSON IO with schema validation and Graphviz DOT export.
- CLI tools for DOT, diagnostics, and validation.
- Argument bundles for higher-level reasoning and credibility propagation scoring.
- Evidence cards and supporting documents for evidence pipelines.
- Deterministic evidence scoring and edge validation helpers (LLM adapters planned).
- Fast deterministic claim/relation tagging plus incremental conversation memory for chat streams.

## Install
```bash
python -m pip install arglib
```

## Quickstart
```python
from arglib.core import ArgumentGraph
from arglib.reasoning import compute_credibility

graph = ArgumentGraph.new(title="Parks")
c1 = graph.add_claim("Green spaces reduce urban heat.", type="fact")
c2 = graph.add_claim("Cities should fund parks.", type="policy")
graph.add_support(c1, c2, rationale="Cooling improves health", gate_mode="OR")

credibility = compute_credibility(graph)
scores = credibility.final_scores
```

## Evidence and scoring
```python
from arglib.ai import score_evidence, validate_edges

scores = score_evidence(graph)
edge_report = validate_edges(graph)
```

## Fast Tagging + Chat Memory
```python
from arglib.ai import ConversationMemory, HybridClaimRelationTagger

tagger = HybridClaimRelationTagger(model_path="models/small_tagger_v1.json")
tagged = tagger.tag("The weather is worsening. Therefore, we should cut emissions.")

memory = ConversationMemory()
memory.ingest_turn("The weather is worsening.", speaker="assistant")
memory.ingest_turn("The weather is not worsening.", speaker="assistant")
```

## Demo UI
```bash
arglib demo-ui --host 127.0.0.1 --port 8765
```

To auto-download model artifacts from Hugging Face at startup:
```bash
arglib demo-ui --auto-download-artifacts --artifact-manifest-path arglib/data/hf_artifacts.json
```

## Axioms
```python
claim = graph.add_claim("Assume baseline demand holds.", is_axiom=True, score=0.6)
warrant = graph.add_warrant("This baseline is reliable.", is_axiom=True, score=0.7)
graph.units[claim].ignore_influence = True
```

## Bundles and credibility propagation
```python
from arglib.reasoning import compute_credibility

bundle = graph.define_argument([c1, c2], bundle_id="arg-1")
cred = compute_credibility(graph)
```

## CLI examples
```bash
arglib dot path/to/graph.json
arglib diagnostics path/to/graph.json --validate
arglib validate path/to/graph.json
```

## Development
This repo uses `uv` for dependency management.
```bash
uv sync
scripts/check.sh
```

## Model Artifact Publishing (Hugging Face)
Keep large checkpoints and training shards out of git and publish them to HF:
```bash
uv run python scripts/publish_hf_artifacts.py \
  --model-repo-id <your-org-or-user>/arglib-artifacts \
  --dataset-repo-id <your-org-or-user>/arglib-datasets
```

This writes `arglib/data/hf_artifacts.json`. To prefetch artifacts:
```bash
uv run python scripts/download_hf_artifacts.py --manifest-path arglib/data/hf_artifacts.json
```

## Documentation
Full docs and guides are available at https://vasanthsarathy.github.io/arglib/.
