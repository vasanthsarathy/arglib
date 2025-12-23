# Repository Guidelines

## Project Structure & Module Organization
Follow the design spec at `specs/argumentation_library_design_spec_batteries_included_python.md` as the source of truth for scope and layout. Current modules live under `arglib/` (core, semantics, io, viz, algorithms, critique, reasoning, cli), tests under `tests/`, and docs under `docs/`. Keep new features aligned to the spec’s layers (core → semantics → critique/AI → integrations).

## Build, Test, and Development Commands
Use `uv` for packaging and environment management:
- `uv sync` to create/update the local environment from `pyproject.toml`.
- `uv run pytest` to run the test suite.
- `uv run ruff check .` for linting.
- `uv run ruff format .` to format code when formatting is enabled.
- `uv run mypy .` for static type checks.
- `uv add <package>` to add runtime dependencies, `uv add --dev <package>` for dev tools.
For a pre-push quality gate, run `scripts/check.sh`. Optional: enable hooks with `git config core.hooksPath .githooks`.

## Release & Publishing Workflow
- Releases are tag-driven: create a GitHub Release (for example `v0.1.0`) to trigger the `Release` workflow and publish to PyPI.
- Trusted publishing is configured on PyPI; no TestPyPI flow is used.
- Docs deploy via GitHub Pages using `.github/workflows/docs.yml` on pushes to `main`.

## Coding Style & Naming Conventions
Target Python conventions for new code:
- Indentation: 4 spaces; avoid tabs.
- Naming: `snake_case` for functions/modules, `CamelCase` for classes, `UPPER_CASE` for constants.
- Types: prefer type hints on public APIs and dataclasses (as shown in the spec).
Follow `ruff` for linting and formatting and `mypy` for type checks.

## Testing Guidelines
Tests live under `tests/` and cover core, semantics, diagnostics, CLI, and schema validation. When adding features:
- Use pytest-style tests named `test_*.py`.
- Keep tests close to the module they cover and favor small, focused fixtures.

## Commit & Pull Request Guidelines
There is no strict convention yet. For new work:
- Use concise, imperative commit subjects (for example "Add ArgumentGraph validators").
- In PRs, include a brief summary, link any related issues, and call out any design-spec changes.

## Roadmap Alignment
This repo is tracking toward the spec milestones:
- v0.1 core: graph model, JSON IO, DOT export, Dung semantics, diagnostics.
- v0.2 ABA: dispute trees + explanations.
- v0.3 AI/mining: ADU extraction, relation mining, pattern bank + matcher.
- v0.4 multimodal evidence.
Keep new work mapped to these milestones, and update the spec if scope shifts.

## Current Focus
Next milestone: v0.2 ABA dispute trees and reasoning explanations. Maintain API stability for v0.1 while extending ABA.

## Specification Updates
Major behavior changes should update the design spec in `specs/argumentation_library_design_spec_batteries_included_python.md` so implementation and documentation stay aligned.
