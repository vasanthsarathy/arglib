# Repository Guidelines

## Project Structure & Module Organization
This repository is currently a design-first skeleton. The only tracked content is `README.md` and the design spec in `specs/argumentation_library_design_spec_batteries_included_python.md`. That spec also proposes the intended Python package layout (for example `arglib/`, `tests/`, `cli/`, and `integrations/`). Treat the spec as the source of truth for where new modules should live until the package is implemented.

## Build, Test, and Development Commands
Use `uv` for packaging and environment management:
- `uv sync` to create/update the local environment from `pyproject.toml`.
- `uv run pytest` to run the test suite.
- `uv run ruff check .` for linting.
- `uv run ruff format .` to format code when formatting is enabled.
- `uv run mypy .` for static type checks.
- `uv add <package>` to add runtime dependencies, `uv add --dev <package>` for dev tools.
For a pre-push quality gate, run `scripts/check.sh`.

## Coding Style & Naming Conventions
Target Python conventions for new code:
- Indentation: 4 spaces; avoid tabs.
- Naming: `snake_case` for functions/modules, `CamelCase` for classes, `UPPER_CASE` for constants.
- Types: prefer type hints on public APIs and dataclasses (as shown in the spec).
Follow `ruff` for linting and formatting and `mypy` for type checks.

## Testing Guidelines
Tests are not present yet, but the spec proposes a `tests/` package with files like `test_core.py` and `test_dung.py`. When tests are added:
- Use pytest-style tests named `test_*.py`.
- Keep tests close to the module they cover and favor small, focused fixtures.

## Commit & Pull Request Guidelines
The Git history currently contains only the initial commit, so no commit convention is established. For new work:
- Use concise, imperative commit subjects (for example "Add ArgumentGraph validators").
- In PRs, include a brief summary, link any related issues, and call out any design-spec changes.

## Specification Updates
Major behavior changes should update the design spec in `specs/argumentation_library_design_spec_batteries_included_python.md` so implementation and documentation stay aligned.
