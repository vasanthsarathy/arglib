# ArgLib v0.1.0 Release Notes (Draft)

## Highlights
- Core argument graph model with JSON IO and schema validation.
- Dung semantics (grounded/preferred/stable/complete) and diagnostics.
- ABA framework with rule-based attack derivation and CLI support.
- Graphviz DOT export and CLI tools.

## Added
- `ArgumentGraph` core dataclasses and helpers.
- JSON serialization (`dumps`/`loads`) + schema validation utilities.
- Diagnostics (cycles, components, SCCs, reachability, degree stats).
- ABA translation to AF with multi-assumption support.
- CLI: `dot`, `diagnostics`, `validate`, `aba`, `version`.
- MkDocs site with Getting Started, API, CLI, v0.1 API surface.

## Tooling
- CI workflow for ruff, mypy, pytest.
- TestPyPI/PyPI publish workflows.

## Known limitations
- ABA dispute trees not implemented yet.
- JSON-LD/AIF/NetworkX export not implemented yet.
- AI mining/pattern detection not implemented yet.
