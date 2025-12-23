# Releases

ArgLib uses GitHub Actions with trusted publishing for TestPyPI and PyPI.

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
