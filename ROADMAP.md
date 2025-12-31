# ArgLib Roadmap

This file mirrors `docs/roadmap.md` for quick reference in the repo root.

## v0.1 (publishable core)
- Core graph model (`ArgumentGraph`, `ArgumentUnit`, `Relation`, evidence, spans).
- JSON IO and schema validation.
- Dung semantics (grounded/preferred/stable/complete) + diagnostics.
- Graphviz DOT export and CLI commands.
- Docs scaffolding and CI/CD.

## v0.2 (ABA + explanations)
- ABA dispute trees and reasoning explanations.
- Evidence scoring and edge validation with confidence.
- Credibility propagation (evidence + network influence).
- Argument bundling: argument = subgraph; bundle graph projection.
Current status: dispute trees, evidence scoring/edge validation, and bundling/credibility propagation complete.

## v0.3 (AI/mining + datasets)
- Argument mining pipeline: ADU extraction, relations, claim types, alignment.
- Assumption generation (implicit premises).
- Pattern bank + matcher + fallacy detection.
- Graph-first dataset generator + linguistic style augmentation.
- Dataset stats and pattern classification utilities.
- Long-document mining: chunking, per-chunk graphs, reconciliation, and graph merging.

## v0.4 (Multimodal + provenance)
- Evidence extraction for PDF/image/video with provenance.
- Supporting document + evidence card pipelines.
- Multimodal alignment into `TextSpan`/evidence attachments.
Current status: supporting documents + evidence cards implemented; multimodal extractors pending.

## v1.0 (stable API + integrations)
- API freeze, docs, tutorials, benchmarks, model cards.
- Integrations (LangChain/LlamaIndex, optional VLM hooks).
- Reporting: critique summaries, evidence quality dashboards.

## v1.1 (UI + service integration)
- `arglib-server`: FastAPI service for graph CRUD, mining, diagnostics, exports.
- `arglib-ui`: web UI for authoring, evidence review, and analysis workflows.
- Shared API contract (OpenAPI) and versioned compatibility with `arglib`.
