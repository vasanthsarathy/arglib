# Releases

ArgLib uses GitHub Actions with trusted publishing for TestPyPI and PyPI.

## Documentation site
Docs deploy via GitHub Pages using the `Docs` workflow on pushes to `main`.

## TestPyPI (manual)
1. Run the `TestPyPI` workflow from GitHub Actions.
2. Install from TestPyPI to validate the package:
```bash
uv run python -m pip install -i https://test.pypi.org/simple/ arglib
```

## PyPI (release)
1. Create a GitHub Release with a tag (for example `v0.1.0`).
2. The `Release` workflow builds and publishes the package to PyPI.

## Versioning
Update the version in `pyproject.toml` before tagging a release.

## v0.1 Release Checklist
1. Update version in `pyproject.toml` and `arglib/__init__.py`.
2. Run `scripts/check.sh`.
3. Run the `TestPyPI` workflow and install from TestPyPI.
4. Tag and create a GitHub Release (for example `v0.1.0`).
