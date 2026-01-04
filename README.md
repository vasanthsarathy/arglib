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

## Documentation
Full docs and guides are available at https://vasanthsarathy.github.io/arglib/.
